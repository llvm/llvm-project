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

            current_paths = self.config.environment.get(variable, "")
            current_paths = current_paths.split(os.path.pathsep)
            paths = [norm(p) for p in current_paths]
            for p in paths_to_add:
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

    def feature_config(self, features, encoding = 'ascii'):
        # Ask llvm-config about the specified feature.
        arguments = [x for (x, _) in features]
        try:
            config_path = os.path.join(self.config.llvm_tools_dir, 'llvm-config')

            llvm_config_cmd = subprocess.Popen(
                [config_path] + arguments,
                stdout = subprocess.PIPE,
                env=self.config.environment)
        except OSError:
            self.lit_config.fatal("Could not find llvm-config in " + self.config.llvm_tools_dir)

        output, _ = llvm_config_cmd.communicate()
        output = output.decode(encoding)
        lines = output.split('\n')
        for (line, (_, patterns)) in zip(lines, features):
            # We should have either a callable or a dictionary.  If it's a
            # dictionary, grep each key against the output and use the value if
            # it matches.  If it's a callable, it does the entire translation.
            if callable(patterns):
                features_to_add = patterns(line)
                self.config.available_features.update(features_to_add)
            else:
                for (match, feature) in patterns.items():
                    if re.search(line, match):
                        self.config.available_features.add(feature)
