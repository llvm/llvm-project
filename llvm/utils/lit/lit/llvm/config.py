import os
import platform
import re
import subprocess
import sys

import lit.util

def binary_feature(on, feature, off_prefix):
    return feature if on else off_prefix + feature

class LLVMConfig(object):

    def __init__(self, lit_config, config):
        self.lit_config = lit_config
        self.config = config

        features = config.available_features

        self.use_lit_shell = False
        # Tweak PATH for Win32 to decide to use bash.exe or not.
        if sys.platform == 'win32':
            # For tests that require Windows to run.
            features.add('system-windows')

            # Seek sane tools in directories and set to $PATH.
            path = self.lit_config.getToolsPath(config.lit_tools_dir,
                                           config.environment['PATH'],
                                           ['cmp.exe', 'grep.exe', 'sed.exe'])
            if path is not None:
                self.with_environment('PATH', path, append_path=True)
            self.use_lit_shell = True

        # Choose between lit's internal shell pipeline runner and a real shell.  If
        # LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
        lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
        if lit_shell_env:
            self.use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

        if not self.use_lit_shell:
            features.add('shell')


        # Running on Darwin OS
        if platform.system() in ['Darwin']:
            # FIXME: lld uses the first, other projects use the second.
            # We should standardize on the former.
            features.add('system-linker-mach-o')
            features.add('system-darwin')
        elif platform.system() in ['Windows']:
            # For tests that require Windows to run.
            features.add('system-windows')

        # Native compilation: host arch == default triple arch
        # Both of these values should probably be in every site config (e.g. as
        # part of the standard header.  But currently they aren't)
        host_triple = getattr(config, 'host_triple', None)
        target_triple = getattr(config, 'target_triple', None)
        if host_triple and host_triple == target_triple:
            features.add("native")

        # Sanitizers.
        sanitizers = getattr(config, 'llvm_use_sanitizer', '')
        sanitizers = frozenset(x.lower() for x in sanitizers.split(';'))
        features.add(binary_feature('address' in sanitizers, 'asan', 'not_'))
        features.add(binary_feature('memory' in sanitizers, 'msan', 'not_'))
        features.add(binary_feature('undefined' in sanitizers, 'ubsan', 'not_'))

        have_zlib = getattr(config, 'have_zlib', None)
        features.add(binary_feature(have_zlib, 'zlib', 'no'))

        # Check if we should run long running tests.
        long_tests = lit_config.params.get("run_long_tests", None)
        if lit.util.pythonize_bool(long_tests):
            features.add("long_tests")

        if target_triple:
            if re.match(r'^x86_64.*-apple', target_triple):
                if 'address' in sanitizers:
                    self.with_environment('ASAN_OPTIONS', 'detect_leaks=1', append_path=True)
            if re.match(r'^x86_64.*-linux', target_triple):
                features.add("x86_64-linux")
            if re.match(r'.*-win32$', target_triple):
                features.add('target-windows')

        use_gmalloc = lit_config.params.get('use_gmalloc', None)
        if lit.util.pythonize_bool(use_gmalloc):
            # Allow use of an explicit path for gmalloc library.
            # Will default to '/usr/lib/libgmalloc.dylib' if not set.
            gmalloc_path_str = lit_config.params.get('gmalloc_path',
                                                     '/usr/lib/libgmalloc.dylib')
            if gmalloc_path_str is not None:
                self.with_environment('DYLD_INSERT_LIBRARIES', gmalloc_path_str)

        breaking_checks = getattr(config, 'enable_abi_breaking_checks', None)
        if lit.util.pythonize_bool(breaking_checks):
            features.add('abi-breaking-checks')

    def with_environment(self, variable, value, append_path = False):
        if append_path:
            # For paths, we should be able to take a list of them and process all
            # of them.
            paths_to_add = value
            if lit.util.is_string(paths_to_add):
                paths_to_add = [paths_to_add]

            def norm(x):
                return os.path.normcase(os.path.normpath(x))

            current_paths = self.config.environment.get(variable, None)
            if current_paths:
                current_paths = current_paths.split(os.path.pathsep)
                paths = [norm(p) for p in current_paths]
            else:
                paths = []

            # If we are passed a list [a b c], then iterating this list forwards
            # and adding each to the beginning would result in b c a.  So we
            # need to iterate in reverse to end up with the original ordering.
            for p in reversed(paths_to_add):
                # Move it to the front if it already exists, otherwise insert it at the
                # beginning.
                p = norm(p)
                try:
                    paths.remove(p)
                except ValueError:
                    pass
                paths = [p] + paths
            value = os.pathsep.join(paths)
        self.config.environment[variable] = value


    def with_system_environment(self, variables, append_path = False):
        if lit.util.is_string(variables):
            variables = [variables]
        for v in variables:
            value = os.environ.get(v)
            if value:
                self.with_environment(v, value, append_path)

    def clear_environment(self, variables):
        for name in variables:
            if name in self.config.environment:
                del self.config.environment[name]

    def get_process_output(self, command):
        try:
            cmd = subprocess.Popen(
                command, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=self.config.environment)
            stdout, stderr = cmd.communicate()
            stdout = lit.util.to_string(stdout)
            stderr = lit.util.to_string(stderr)
            return (stdout, stderr)
        except OSError:
            self.lit_config.fatal("Could not run process %s" % command)

    def feature_config(self, features):
        # Ask llvm-config about the specified feature.
        arguments = [x for (x, _) in features]
        config_path = os.path.join(self.config.llvm_tools_dir, 'llvm-config')

        output, _ = self.get_process_output([config_path] + arguments)
        lines = output.split('\n')

        for (feature_line, (_, patterns)) in zip(lines, features):
            # We should have either a callable or a dictionary.  If it's a
            # dictionary, grep each key against the output and use the value if
            # it matches.  If it's a callable, it does the entire translation.
            if callable(patterns):
                features_to_add = patterns(feature_line)
                self.config.available_features.update(features_to_add)
            else:
                for (re_pattern, feature) in patterns.items():
                    if re.search(re_pattern, feature_line):
                        self.config.available_features.add(feature)


    # Note that when substituting %clang_cc1 also fill in the include directory of
    # the builtin headers. Those are part of even a freestanding environment, but
    # Clang relies on the driver to locate them.
    def get_clang_builtin_include_dir(self, clang):
        # FIXME: Rather than just getting the version, we should have clang print
        # out its resource dir here in an easy to scrape form.
        clang_dir, _ = self.get_process_output([clang, '-print-file-name=include'])

        if not clang_dir:
          self.lit_config.fatal("Couldn't find the include dir for Clang ('%s')" % clang)

        clang_dir = clang_dir.strip()
        if sys.platform in ['win32'] and not self.use_lit_shell:
            # Don't pass dosish path separator to msys bash.exe.
            clang_dir = clang_dir.replace('\\', '/')
        # Ensure the result is an ascii string, across Python2.5+ - Python3.
        return clang_dir

    def make_itanium_abi_triple(self, triple):
        m = re.match(r'(\w+)-(\w+)-(\w+)', triple)
        if not m:
          self.lit_config.fatal("Could not turn '%s' into Itanium ABI triple" % triple)
        if m.group(3).lower() != 'win32':
          # All non-win32 triples use the Itanium ABI.
          return triple
        return m.group(1) + '-' + m.group(2) + '-mingw32'

    def make_msabi_triple(self, triple):
        m = re.match(r'(\w+)-(\w+)-(\w+)', triple)
        if not m:
          self.lit_config.fatal("Could not turn '%s' into MS ABI triple" % triple)
        isa = m.group(1).lower()
        vendor = m.group(2).lower()
        os = m.group(3).lower()
        if os == 'win32':
          # If the OS is win32, we're done.
          return triple
        if isa.startswith('x86') or isa == 'amd64' or re.match(r'i\d86', isa):
          # For x86 ISAs, adjust the OS.
          return isa + '-' + vendor + '-win32'
        # -win32 is not supported for non-x86 targets; use a default.
        return 'i686-pc-win32'

    def add_tool_substitutions(self, tools, search_dirs, warn_missing = True):
        if lit.util.is_string(search_dirs):
            search_dirs = [search_dirs]

        search_dirs = os.pathsep.join(search_dirs)
        for tool in tools:
            # Extract the tool name from the pattern.  This relies on the tool
            # name being surrounded by \b word match operators.  If the
            # pattern starts with "| ", include it in the string to be
            # substituted.
            if lit.util.is_string(tool):
                tool = lit.util.make_word_regex(tool)
            else:
                tool = str(tool)

            tool_match = re.match(r"^(\\)?((\| )?)\W+b([0-9A-Za-z-_\.]+)\\b\W*$",
                                  tool)
            if not tool_match:
                continue

            tool_pipe = tool_match.group(2)
            tool_name = tool_match.group(4)
            tool_path = lit.util.which(tool_name, search_dirs)
            if not tool_path:
                if warn_missing:
                    # Warn, but still provide a substitution.
                    self.lit_config.note('Did not find ' + tool_name + ' in %s' % search_dirs)
                tool_path = self.config.llvm_tools_dir + '/' + tool_name

            if tool_name == 'llc' and os.environ.get('LLVM_ENABLE_MACHINE_VERIFIER') == '1':
                tool_path += ' -verify-machineinstrs'
            if tool_name == 'llvm-go':
                exe = getattr(self.config, 'go_executable', None)
                if exe:
                    tool_path += " go=" + exe

            self.config.substitutions.append((tool, tool_pipe + tool_path))
