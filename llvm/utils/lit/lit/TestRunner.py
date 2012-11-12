import os, signal, subprocess, sys
import StringIO

import ShUtil
import Test
import Util

import platform
import tempfile

import re

class InternalShellError(Exception):
    def __init__(self, command, message):
        self.command = command
        self.message = message

kIsWindows = platform.system() == 'Windows'

# Don't use close_fds on Windows.
kUseCloseFDs = not kIsWindows

# Use temporary files to replace /dev/null on Windows.
kAvoidDevNull = kIsWindows

def executeCommand(command, cwd=None, env=None):
    # Close extra file handles on UNIX (on Windows this cannot be done while
    # also redirecting input).
    close_fds = not kIsWindows

    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=env, close_fds=close_fds)
    out,err = p.communicate()
    exitCode = p.wait()

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    return out, err, exitCode

def executeShCmd(cmd, cfg, cwd, results):
    if isinstance(cmd, ShUtil.Seq):
        if cmd.op == ';':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            return executeShCmd(cmd.rhs, cfg, cwd, results)

        if cmd.op == '&':
            raise NotImplementedError,"unsupported test command: '&'"

        if cmd.op == '||':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            if res != 0:
                res = executeShCmd(cmd.rhs, cfg, cwd, results)
            return res
        if cmd.op == '&&':
            res = executeShCmd(cmd.lhs, cfg, cwd, results)
            if res is None:
                return res

            if res == 0:
                res = executeShCmd(cmd.rhs, cfg, cwd, results)
            return res

        raise ValueError,'Unknown shell command: %r' % cmd.op

    assert isinstance(cmd, ShUtil.Pipeline)
    procs = []
    input = subprocess.PIPE
    stderrTempFiles = []
    opened_files = []
    named_temp_files = []
    # To avoid deadlock, we use a single stderr stream for piped
    # output. This is null until we have seen some output using
    # stderr.
    for i,j in enumerate(cmd.commands):
        # Apply the redirections, we use (N,) as a sentinal to indicate stdin,
        # stdout, stderr for N equal to 0, 1, or 2 respectively. Redirects to or
        # from a file are represented with a list [file, mode, file-object]
        # where file-object is initially None.
        redirects = [(0,), (1,), (2,)]
        for r in j.redirects:
            if r[0] == ('>',2):
                redirects[2] = [r[1], 'w', None]
            elif r[0] == ('>>',2):
                redirects[2] = [r[1], 'a', None]
            elif r[0] == ('>&',2) and r[1] in '012':
                redirects[2] = redirects[int(r[1])]
            elif r[0] == ('>&',) or r[0] == ('&>',):
                redirects[1] = redirects[2] = [r[1], 'w', None]
            elif r[0] == ('>',):
                redirects[1] = [r[1], 'w', None]
            elif r[0] == ('>>',):
                redirects[1] = [r[1], 'a', None]
            elif r[0] == ('<',):
                redirects[0] = [r[1], 'r', None]
            else:
                raise NotImplementedError,"Unsupported redirect: %r" % (r,)

        # Map from the final redirections to something subprocess can handle.
        final_redirects = []
        for index,r in enumerate(redirects):
            if r == (0,):
                result = input
            elif r == (1,):
                if index == 0:
                    raise NotImplementedError,"Unsupported redirect for stdin"
                elif index == 1:
                    result = subprocess.PIPE
                else:
                    result = subprocess.STDOUT
            elif r == (2,):
                if index != 2:
                    raise NotImplementedError,"Unsupported redirect on stdout"
                result = subprocess.PIPE
            else:
                if r[2] is None:
                    if kAvoidDevNull and r[0] == '/dev/null':
                        r[2] = tempfile.TemporaryFile(mode=r[1])
                    else:
                        r[2] = open(r[0], r[1])
                    # Workaround a Win32 and/or subprocess bug when appending.
                    #
                    # FIXME: Actually, this is probably an instance of PR6753.
                    if r[1] == 'a':
                        r[2].seek(0, 2)
                    opened_files.append(r[2])
                result = r[2]
            final_redirects.append(result)

        stdin, stdout, stderr = final_redirects

        # If stderr wants to come from stdout, but stdout isn't a pipe, then put
        # stderr on a pipe and treat it as stdout.
        if (stderr == subprocess.STDOUT and stdout != subprocess.PIPE):
            stderr = subprocess.PIPE
            stderrIsStdout = True
        else:
            stderrIsStdout = False

            # Don't allow stderr on a PIPE except for the last
            # process, this could deadlock.
            #
            # FIXME: This is slow, but so is deadlock.
            if stderr == subprocess.PIPE and j != cmd.commands[-1]:
                stderr = tempfile.TemporaryFile(mode='w+b')
                stderrTempFiles.append((i, stderr))

        # Resolve the executable path ourselves.
        args = list(j.args)
        args[0] = Util.which(args[0], cfg.environment['PATH'])
        if not args[0]:
            raise InternalShellError(j, '%r: command not found' % j.args[0])

        # Replace uses of /dev/null with temporary files.
        if kAvoidDevNull:
            for i,arg in enumerate(args):
                if arg == "/dev/null":
                    f = tempfile.NamedTemporaryFile(delete=False)
                    f.close()
                    named_temp_files.append(f.name)
                    args[i] = f.name

        procs.append(subprocess.Popen(args, cwd=cwd,
                                      stdin = stdin,
                                      stdout = stdout,
                                      stderr = stderr,
                                      env = cfg.environment,
                                      close_fds = kUseCloseFDs))

        # Immediately close stdin for any process taking stdin from us.
        if stdin == subprocess.PIPE:
            procs[-1].stdin.close()
            procs[-1].stdin = None

        # Update the current stdin source.
        if stdout == subprocess.PIPE:
            input = procs[-1].stdout
        elif stderrIsStdout:
            input = procs[-1].stderr
        else:
            input = subprocess.PIPE

    # Explicitly close any redirected files. We need to do this now because we
    # need to release any handles we may have on the temporary files (important
    # on Win32, for example). Since we have already spawned the subprocess, our
    # handles have already been transferred so we do not need them anymore.
    for f in opened_files:
        f.close()

    # FIXME: There is probably still deadlock potential here. Yawn.
    procData = [None] * len(procs)
    procData[-1] = procs[-1].communicate()

    for i in range(len(procs) - 1):
        if procs[i].stdout is not None:
            out = procs[i].stdout.read()
        else:
            out = ''
        if procs[i].stderr is not None:
            err = procs[i].stderr.read()
        else:
            err = ''
        procData[i] = (out,err)

    # Read stderr out of the temp files.
    for i,f in stderrTempFiles:
        f.seek(0, 0)
        procData[i] = (procData[i][0], f.read())

    exitCode = None
    for i,(out,err) in enumerate(procData):
        res = procs[i].wait()
        # Detect Ctrl-C in subprocess.
        if res == -signal.SIGINT:
            raise KeyboardInterrupt

        results.append((cmd.commands[i], out, err, res))
        if cmd.pipe_err:
            # Python treats the exit code as a signed char.
            if res < 0:
                exitCode = min(exitCode, res)
            else:
                exitCode = max(exitCode, res)
        else:
            exitCode = res

    # Remove any named temporary files we created.
    for f in named_temp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    if cmd.negate:
        exitCode = not exitCode

    return exitCode

def executeScriptInternal(test, litConfig, tmpBase, commands, cwd):
    ln = ' &&\n'.join(commands)
    try:
        cmd = ShUtil.ShParser(ln, litConfig.isWindows).parse()
    except:
        return (Test.FAIL, "shell parser error on: %r" % ln)

    results = []
    try:
        exitCode = executeShCmd(cmd, test.config, cwd, results)
    except InternalShellError,e:
        out = ''
        err = e.message
        exitCode = 255

    out = err = ''
    for i,(cmd, cmd_out,cmd_err,res) in enumerate(results):
        out += 'Command %d: %s\n' % (i, ' '.join('"%s"' % s for s in cmd.args))
        out += 'Command %d Result: %r\n' % (i, res)
        out += 'Command %d Output:\n%s\n\n' % (i, cmd_out)
        out += 'Command %d Stderr:\n%s\n\n' % (i, cmd_err)

    return out, err, exitCode

def executeTclScriptInternal(test, litConfig, tmpBase, commands, cwd):
    import TclUtil
    cmds = []
    for ln in commands:
        # Given the unfortunate way LLVM's test are written, the line gets
        # backslash substitution done twice.
        ln = TclUtil.TclLexer(ln).lex_unquoted(process_all = True)

        try:
            tokens = list(TclUtil.TclLexer(ln).lex())
        except:
            return (Test.FAIL, "Tcl lexer error on: %r" % ln)

        # Validate there are no control tokens.
        for t in tokens:
            if not isinstance(t, str):
                return (Test.FAIL,
                        "Invalid test line: %r containing %r" % (ln, t))

        try:
            cmds.append(TclUtil.TclExecCommand(tokens).parse_pipeline())
        except:
            return (Test.FAIL, "Tcl 'exec' parse error on: %r" % ln)

    if litConfig.useValgrind:
        for pipeline in cmds:
            if pipeline.commands:
                # Only valgrind the first command in each pipeline, to avoid
                # valgrinding things like grep, not, and FileCheck.
                cmd = pipeline.commands[0]
                cmd.args = litConfig.valgrindArgs + cmd.args

    cmd = cmds[0]
    for c in cmds[1:]:
        cmd = ShUtil.Seq(cmd, '&&', c)

    # FIXME: This is lame, we shouldn't need bash. See PR5240.
    bashPath = litConfig.getBashPath()
    if litConfig.useTclAsSh and bashPath:
        script = tmpBase + '.script'

        # Write script file
        f = open(script,'w')
        print >>f, 'set -o pipefail'
        cmd.toShell(f, pipefail = True)
        f.close()

        if 0:
            print >>sys.stdout, cmd
            print >>sys.stdout, open(script).read()
            print >>sys.stdout
            return '', '', 0

        command = [litConfig.getBashPath(), script]
        out,err,exitCode = executeCommand(command, cwd=cwd,
                                          env=test.config.environment)

        return out,err,exitCode
    else:
        results = []
        try:
            exitCode = executeShCmd(cmd, test.config, cwd, results)
        except InternalShellError,e:
            results.append((e.command, '', e.message + '\n', 255))
            exitCode = 255

    out = err = ''

    for i,(cmd, cmd_out, cmd_err, res) in enumerate(results):
        out += 'Command %d: %s\n' % (i, ' '.join('"%s"' % s for s in cmd.args))
        out += 'Command %d Result: %r\n' % (i, res)
        out += 'Command %d Output:\n%s\n\n' % (i, cmd_out)
        out += 'Command %d Stderr:\n%s\n\n' % (i, cmd_err)

    return out, err, exitCode

def executeScript(test, litConfig, tmpBase, commands, cwd):
    bashPath = litConfig.getBashPath();
    isWin32CMDEXE = (litConfig.isWindows and not bashPath)
    script = tmpBase + '.script'
    if isWin32CMDEXE:
        script += '.bat'

    # Write script file
    f = open(script,'w')
    if isWin32CMDEXE:
        f.write('\nif %ERRORLEVEL% NEQ 0 EXIT\n'.join(commands))
    else:
        f.write(' &&\n'.join(commands))
    f.write('\n')
    f.close()

    if isWin32CMDEXE:
        command = ['cmd','/c', script]
    else:
        if bashPath:
            command = [bashPath, script]
        else:
            command = ['/bin/sh', script]
        if litConfig.useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # run on clang with no real loss.
            command = litConfig.valgrindArgs + command

    return executeCommand(command, cwd=cwd, env=test.config.environment)

def isExpectedFail(test, xfails):
    # Check if any of the xfails match an available feature or the target.
    for item in xfails:
        # If this is the wildcard, it always fails.
        if item == '*':
            return True

        # If this is an exact match for one of the features, it fails.
        if item in test.config.available_features:
            return True

        # If this is a part of the target triple, it fails.
        if item in test.suite.config.target_triple:
            return True

    return False

def parseIntegratedTestScript(test, normalize_slashes=False,
                              extra_substitutions=[]):
    """parseIntegratedTestScript - Scan an LLVM/Clang style integrated test
    script and extract the lines to 'RUN' as well as 'XFAIL' and 'REQUIRES'
    information. The RUN lines also will have variable substitution performed.
    """

    # Get the temporary location, this is always relative to the test suite
    # root, not test source root.
    #
    # FIXME: This should not be here?
    sourcepath = test.getSourcePath()
    sourcedir = os.path.dirname(sourcepath)
    execpath = test.getExecPath()
    execdir,execbase = os.path.split(execpath)
    tmpDir = os.path.join(execdir, 'Output')
    tmpBase = os.path.join(tmpDir, execbase)
    if test.index is not None:
        tmpBase += '_%d' % test.index

    # Normalize slashes, if requested.
    if normalize_slashes:
        sourcepath = sourcepath.replace('\\', '/')
        sourcedir = sourcedir.replace('\\', '/')
        tmpDir = tmpDir.replace('\\', '/')
        tmpBase = tmpBase.replace('\\', '/')

    # We use #_MARKER_# to hide %% while we do the other substitutions.
    substitutions = list(extra_substitutions)
    substitutions.extend([('%%', '#_MARKER_#')])
    substitutions.extend(test.config.substitutions)
    substitutions.extend([('%s', sourcepath),
                          ('%S', sourcedir),
                          ('%p', sourcedir),
                          ('%{pathsep}', os.pathsep),
                          ('%t', tmpBase + '.tmp'),
                          ('%T', tmpDir),
                          # FIXME: Remove this once we kill DejaGNU.
                          ('%abs_tmp', tmpBase + '.tmp'),
                          ('#_MARKER_#', '%')])

    # Collect the test lines from the script.
    script = []
    xfails = []
    requires = []
    for ln in open(sourcepath):
        if 'RUN:' in ln:
            # Isolate the command to run.
            index = ln.index('RUN:')
            ln = ln[index+4:]

            # Trim trailing whitespace.
            ln = ln.rstrip()

            # Collapse lines with trailing '\\'.
            if script and script[-1][-1] == '\\':
                script[-1] = script[-1][:-1] + ln
            else:
                script.append(ln)
        elif 'XFAIL:' in ln:
            items = ln[ln.index('XFAIL:') + 6:].split(',')
            xfails.extend([s.strip() for s in items])
        elif 'REQUIRES:' in ln:
            items = ln[ln.index('REQUIRES:') + 9:].split(',')
            requires.extend([s.strip() for s in items])
        elif 'END.' in ln:
            # Check for END. lines.
            if ln[ln.index('END.'):].strip() == 'END.':
                break

    # Apply substitutions to the script.  Allow full regular
    # expression syntax.  Replace each matching occurrence of regular
    # expression pattern a with substitution b in line ln.
    def processLine(ln):
        # Apply substitutions
        for a,b in substitutions:
            if kIsWindows:
                b = b.replace("\\","\\\\")
            ln = re.sub(a, b, ln)

        # Strip the trailing newline and any extra whitespace.
        return ln.strip()
    script = map(processLine, script)

    # Verify the script contains a run line.
    if not script:
        return (Test.UNRESOLVED, "Test has no run line!")

    # Check for unterminated run lines.
    if script[-1][-1] == '\\':
        return (Test.UNRESOLVED, "Test has unterminated run lines (with '\\')")

    # Check that we have the required features:
    missing_required_features = [f for f in requires
                                 if f not in test.config.available_features]
    if missing_required_features:
        msg = ', '.join(missing_required_features)
        return (Test.UNSUPPORTED,
                "Test requires the following features: %s" % msg)

    isXFail = isExpectedFail(test, xfails)
    return script,isXFail,tmpBase,execdir

def formatTestOutput(status, out, err, exitCode, failDueToStderr, script):
    output = StringIO.StringIO()
    print >>output, "Script:"
    print >>output, "--"
    print >>output, '\n'.join(script)
    print >>output, "--"
    print >>output, "Exit Code: %r" % exitCode,
    if failDueToStderr:
        print >>output, "(but there was output on stderr)"
    else:
        print >>output
    if out:
        print >>output, "Command Output (stdout):"
        print >>output, "--"
        output.write(out)
        print >>output, "--"
    if err:
        print >>output, "Command Output (stderr):"
        print >>output, "--"
        output.write(err)
        print >>output, "--"
    return (status, output.getvalue())

def executeTclTest(test, litConfig):
    if test.config.unsupported:
        return (Test.UNSUPPORTED, 'Test is unsupported')

    # Parse the test script, normalizing slashes in substitutions on Windows
    # (since otherwise Tcl style lexing will treat them as escapes).
    res = parseIntegratedTestScript(test, normalize_slashes=kIsWindows)
    if len(res) == 2:
        return res

    script, isXFail, tmpBase, execdir = res

    if litConfig.noExecute:
        return (Test.PASS, '')

    # Create the output directory if it does not already exist.
    Util.mkdir_p(os.path.dirname(tmpBase))

    res = executeTclScriptInternal(test, litConfig, tmpBase, script, execdir)
    if len(res) == 2:
        return res

    # Test for failure. In addition to the exit code, Tcl commands are
    # considered to fail if there is any standard error output.
    out,err,exitCode = res
    if isXFail:
        ok = exitCode != 0 or err and not litConfig.ignoreStdErr
        if ok:
            status = Test.XFAIL
        else:
            status = Test.XPASS
    else:
        ok = exitCode == 0 and (not err or litConfig.ignoreStdErr)
        if ok:
            status = Test.PASS
        else:
            status = Test.FAIL

    if ok:
        return (status,'')

    # Set a flag for formatTestOutput so it can explain why the test was
    # considered to have failed, despite having an exit code of 0.
    failDueToStderr = exitCode == 0 and err and not litConfig.ignoreStdErr

    return formatTestOutput(status, out, err, exitCode, failDueToStderr, script)

def executeShTest(test, litConfig, useExternalSh,
                  extra_substitutions=[]):
    if test.config.unsupported:
        return (Test.UNSUPPORTED, 'Test is unsupported')

    res = parseIntegratedTestScript(test, useExternalSh, extra_substitutions)
    if len(res) == 2:
        return res

    script, isXFail, tmpBase, execdir = res

    if litConfig.noExecute:
        return (Test.PASS, '')

    # Create the output directory if it does not already exist.
    Util.mkdir_p(os.path.dirname(tmpBase))

    if useExternalSh:
        res = executeScript(test, litConfig, tmpBase, script, execdir)
    else:
        res = executeScriptInternal(test, litConfig, tmpBase, script, execdir)
    if len(res) == 2:
        return res

    out,err,exitCode = res
    if isXFail:
        ok = exitCode != 0
        if ok:
            status = Test.XFAIL
        else:
            status = Test.XPASS
    else:
        ok = exitCode == 0
        if ok:
            status = Test.PASS
        else:
            status = Test.FAIL

    if ok:
        return (status,'')

    # Sh tests are not considered to fail just from stderr output.
    failDueToStderr = False

    return formatTestOutput(status, out, err, exitCode, failDueToStderr, script)
