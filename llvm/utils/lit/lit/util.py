import errno
import itertools
import math
import os
import platform
import signal
import subprocess
import sys

def to_bytes(str):
    # Encode to UTF-8 to get binary data.
    return str.encode('utf-8')

def to_string(bytes):
    if isinstance(bytes, str):
        return bytes
    return to_bytes(bytes)

def convert_string(bytes):
    try:
        return to_string(bytes.decode('utf-8'))
    except UnicodeError:
        return str(bytes)

def detectCPUs():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(capture(['sysctl', '-n', 'hw.ncpu']))
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1 # Default

def mkdir_p(path):
    """mkdir_p(path) - Make the "path" directory, if it does not exist; this
    will also make directories for any missing parent directories."""
    if not path or os.path.exists(path):
        return

    parent = os.path.dirname(path)
    if parent != path:
        mkdir_p(parent)

    try:
        os.mkdir(path)
    except OSError:
        e = sys.exc_info()[1]
        # Ignore EEXIST, which may occur during a race condition.
        if e.errno != errno.EEXIST:
            raise

def capture(args, env=None):
    """capture(command) - Run the given command (or argv list) in a shell and
    return the standard output."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env)
    out,_ = p.communicate()
    return convert_string(out)

def which(command, paths = None):
    """which(command, [paths]) - Look up the given command in the paths string
    (or the PATH environment variable, if unspecified)."""

    if paths is None:
        paths = os.environ.get('PATH','')

    # Check for absolute match first.
    if os.path.isfile(command):
        return command

    # Would be nice if Python had a lib function for this.
    if not paths:
        paths = os.defpath

    # Get suffixes to search.
    # On Cygwin, 'PATHEXT' may exist but it should not be used.
    if os.pathsep == ';':
        pathext = os.environ.get('PATHEXT', '').split(';')
    else:
        pathext = ['']

    # Search the paths...
    for path in paths.split(os.pathsep):
        for ext in pathext:
            p = os.path.join(path, command + ext)
            if os.path.exists(p):
                return p

    return None

def checkToolsPath(dir, tools):
    for tool in tools:
        if not os.path.exists(os.path.join(dir, tool)):
            return False;
    return True;

def whichTools(tools, paths):
    for path in paths.split(os.pathsep):
        if checkToolsPath(path, tools):
            return path
    return None

def printHistogram(items, title = 'Items'):
    items.sort(key = lambda item: item[1])

    maxValue = max([v for _,v in items])

    # Select first "nice" bar height that produces more than 10 bars.
    power = int(math.ceil(math.log(maxValue, 10)))
    for inc in itertools.cycle((5, 2, 2.5, 1)):
        barH = inc * 10**power
        N = int(math.ceil(maxValue / barH))
        if N > 10:
            break
        elif inc == 1:
            power -= 1

    histo = [set() for i in range(N)]
    for name,v in items:
        bin = min(int(N * v/maxValue), N-1)
        histo[bin].add(name)

    barW = 40
    hr = '-' * (barW + 34)
    print('\nSlowest %s:' % title)
    print(hr)
    for name,value in items[-20:]:
        print('%.2fs: %s' % (value, name))
    print('\n%s Times:' % title)
    print(hr)
    pDigits = int(math.ceil(math.log(maxValue, 10)))
    pfDigits = max(0, 3-pDigits)
    if pfDigits:
        pDigits += pfDigits + 1
    cDigits = int(math.ceil(math.log(len(items), 10)))
    print("[%s] :: [%s] :: [%s]" % ('Range'.center((pDigits+1)*2 + 3),
                                    'Percentage'.center(barW),
                                    'Count'.center(cDigits*2 + 1)))
    print(hr)
    for i,row in enumerate(histo):
        pct = float(len(row)) / len(items)
        w = int(barW * pct)
        print("[%*.*fs,%*.*fs) :: [%s%s] :: [%*d/%*d]" % (
            pDigits, pfDigits, i*barH, pDigits, pfDigits, (i+1)*barH,
            '*'*w, ' '*(barW-w), cDigits, len(row), cDigits, len(items)))

# Close extra file handles on UNIX (on Windows this cannot be done while
# also redirecting input).
kUseCloseFDs = not (platform.system() == 'Windows')
def executeCommand(command, cwd=None, env=None, input=None):
    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=env, close_fds=kUseCloseFDs)
    out,err = p.communicate(input=input)
    exitCode = p.wait()

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    # Ensure the resulting output is always of string type.
    out = convert_string(out)
    err = convert_string(err)

    return out, err, exitCode

def usePlatformSdkOnDarwin(config, lit_config):
    # On Darwin, support relocatable SDKs by providing Clang with a
    # default system root path.
    if 'darwin' in config.target_triple:
        try:
            cmd = subprocess.Popen(['xcrun', '--show-sdk-path'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = cmd.communicate()
            out = out.strip()
            res = cmd.wait()
        except OSError:
            res = -1
        if res == 0 and out:
            sdk_path = out
            lit_config.note('using SDKROOT: %r' % sdk_path)
            config.environment['SDKROOT'] = sdk_path
