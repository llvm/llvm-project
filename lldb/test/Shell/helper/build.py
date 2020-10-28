#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import signal
import subprocess
import sys

if sys.platform == 'win32':
    # This module was renamed in Python 3.  Make sure to import it using a
    # consistent name regardless of python version.
    try:
        import winreg
    except:
        import _winreg as winreg

if __name__ != "__main__":
    raise RuntimeError("Do not import this script, run it instead")


parser = argparse.ArgumentParser(description='LLDB compilation wrapper')
parser.add_argument('--arch',
                    metavar='arch',
                    dest='arch',
                    required=True,
                    default='host',
                    choices=['32', '64', 'host'],
                    help='Specify the architecture to target.')

parser.add_argument('--compiler',
                    metavar='compiler',
                    dest='compiler',
                    required=True,
                    help='Path to a compiler executable, or one of the values [any, msvc, clang-cl, gcc, clang]')

parser.add_argument('--libs-dir',
                    metavar='directory',
                    dest='libs_dir',
                    required=False,
                    action='append',
                    help='If specified, a path to linked libraries to be passed via -L')

parser.add_argument('--tools-dir',
                    metavar='directory',
                    dest='tools_dir',
                    required=False,
                    action='append',
                    help='If specified, a path to search in addition to PATH when --compiler is not an exact path')

if sys.platform == 'darwin':
    parser.add_argument('--apple-sdk',
                        metavar='apple_sdk',
                        dest='apple_sdk',
                        default="macosx",
                        help='Specify the name of the Apple SDK (macosx, macosx.internal, iphoneos, iphoneos.internal, or path to SDK) and use the appropriate tools from that SDK\'s toolchain.')

parser.add_argument('--output', '-o',
                    dest='output',
                    metavar='file',
                    required=False,
                    default='',
                    help='Path to output file')

parser.add_argument('--outdir', '-d',
                    dest='outdir',
                    metavar='directory',
                    required=False,
                    help='Directory for output files')

parser.add_argument('--nodefaultlib',
                    dest='nodefaultlib',
                    action='store_true',
                    default=False,
                    help='When specified, the resulting image should not link against system libraries or include system headers.  Useful when writing cross-targeting tests.')

parser.add_argument('--opt',
                    dest='opt',
                    default='none',
                    choices=['none', 'basic', 'lto'],
                    help='Optimization level')

parser.add_argument('--mode',
                    dest='mode',
                    default='compile-and-link',
                    choices=['compile', 'link', 'compile-and-link'],
                    help='Specifies whether to compile, link, or both')

parser.add_argument('--noclean',
                    dest='clean',
                    action='store_false',
                    default=True,
                    help='Dont clean output file before building')

parser.add_argument('--verbose',
                    dest='verbose',
                    action='store_true',
                    default=False,
                    help='Print verbose output')

parser.add_argument('-n', '--dry-run',
                    dest='dry',
                    action='store_true',
                    default=False,
                    help='Print the commands that would run, but dont actually run them')

parser.add_argument('inputs',
                    metavar='file',
                    nargs='+',
                    help='Source file(s) to compile / object file(s) to link')


args = parser.parse_args(args=sys.argv[1:])


def to_string(b):
    """Return the parameter as type 'str', possibly encoding it.

    In Python2, the 'str' type is the same as 'bytes'. In Python3, the
    'str' type is (essentially) Python2's 'unicode' type, and 'bytes' is
    distinct.

    This function is copied from llvm/utils/lit/lit/util.py
    """
    if isinstance(b, str):
        # In Python2, this branch is taken for types 'str' and 'bytes'.
        # In Python3, this branch is taken only for 'str'.
        return b
    if isinstance(b, bytes):
        # In Python2, this branch is never taken ('bytes' is handled as 'str').
        # In Python3, this is true only for 'bytes'.
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            # If the value is not valid Unicode, return the default
            # repr-line encoding.
            return str(b)

    # By this point, here's what we *don't* have:
    #
    #  - In Python2:
    #    - 'str' or 'bytes' (1st branch above)
    #  - In Python3:
    #    - 'str' (1st branch above)
    #    - 'bytes' (2nd branch above)
    #
    # The last type we might expect is the Python2 'unicode' type. There is no
    # 'unicode' type in Python3 (all the Python3 cases were already handled). In
    # order to get a 'str' object, we need to encode the 'unicode' object.
    try:
        return b.encode('utf-8')
    except AttributeError:
        raise TypeError('not sure how to convert %s to %s' % (type(b), str))

def format_text(lines, indent_0, indent_n):
    result = ' ' * indent_0 + lines[0]
    for next in lines[1:]:
        result = result + '\n{0}{1}'.format(' ' * indent_n, next)
    return result

def print_environment(env):
    if env is None:
        print('    Inherited')
        return
    for e in env:
        value = env[e]
        lines = value.split(os.pathsep)
        formatted_value = format_text(lines, 0, 7 + len(e))
        print('    {0} = {1}'.format(e, formatted_value))

def find_executable(binary_name, search_paths):
    if sys.platform == 'win32':
        binary_name = binary_name + '.exe'

    search_paths = os.pathsep.join(search_paths)
    paths = search_paths + os.pathsep + os.environ.get('PATH', '')
    for path in paths.split(os.pathsep):
        p = os.path.join(path, binary_name)
        if os.path.exists(p) and not os.path.isdir(p):
            return os.path.normpath(p)
    return None

def find_toolchain(compiler, tools_dir):
    if compiler == 'msvc':
        return ('msvc', find_executable('cl', tools_dir))
    if compiler == 'clang-cl':
        return ('clang-cl', find_executable('clang-cl', tools_dir))
    if compiler == 'gcc':
        return ('gcc', find_executable('g++', tools_dir))
    if compiler == 'clang':
        return ('clang', find_executable('clang++', tools_dir))
    if compiler == 'any':
        priorities = []
        if sys.platform == 'win32':
            priorities = ['clang-cl', 'msvc', 'clang', 'gcc']
        else:
            priorities = ['clang', 'gcc', 'clang-cl']
        for toolchain in priorities:
            (type, dir) = find_toolchain(toolchain, tools_dir)
            if type and dir:
                return (type, dir)
        # Could not find any toolchain.
        return (None, None)

    # From here on, assume that |compiler| is a path to a file.
    file = os.path.basename(compiler)
    name, ext = os.path.splitext(file)
    if file.lower() == 'cl.exe':
        return ('msvc', compiler)
    if name == 'clang-cl':
        return ('clang-cl', compiler)
    if name.startswith('clang'):
        return ('clang', compiler)
    if name.startswith('gcc') or name.startswith('g++'):
        return ('gcc', compiler)
    if name == 'cc' or name == 'c++':
        return ('generic', compiler)
    return ('unknown', compiler)

class Builder(object):
    def __init__(self, toolchain_type, args, obj_ext):
        self.toolchain_type = toolchain_type
        self.inputs = args.inputs
        self.arch = args.arch
        self.opt = args.opt
        self.outdir = args.outdir
        self.compiler = args.compiler
        self.clean = args.clean
        self.output = args.output
        self.mode = args.mode
        self.nodefaultlib = args.nodefaultlib
        self.verbose = args.verbose
        self.obj_ext = obj_ext
        self.lib_paths = args.libs_dir

    def _exe_file_name(self):
        assert self.mode != 'compile'
        return self.output

    def _output_name(self, input, extension, with_executable=False):
        basename = os.path.splitext(os.path.basename(input))[0] + extension
        if with_executable:
            exe_basename = os.path.basename(self._exe_file_name())
            basename = exe_basename + '-' + basename

        output = os.path.join(self.outdir, basename)
        return os.path.normpath(output)

    def _obj_file_names(self):
        if self.mode == 'link':
            return self.inputs

        if self.mode == 'compile-and-link':
            # Object file names should factor in both the input file (source)
            # name and output file (executable) name, to ensure that two tests
            # which share a common source file don't race to write the same
            # object file.
            return [self._output_name(x, self.obj_ext, True) for x in self.inputs]

        if self.mode == 'compile' and self.output:
            return [self.output]

        return [self._output_name(x, self.obj_ext) for x in self.inputs]

    def build_commands(self):
        commands = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            for input, output in zip(self.inputs, self._obj_file_names()):
                commands.append(self._get_compilation_command(input, output))
        if self.mode == 'link' or self.mode == 'compile-and-link':
            commands.append(self._get_link_command())
        return commands


class MsvcBuilder(Builder):
    def __init__(self, toolchain_type, args):
        Builder.__init__(self, toolchain_type, args, '.obj')

        self.msvc_arch_str = 'x86' if self.arch == '32' else 'x64'

        if toolchain_type == 'msvc':
            # Make sure we're using the appropriate toolchain for the desired
            # target type.
            compiler_parent_dir = os.path.dirname(self.compiler)
            selected_target_version = os.path.basename(compiler_parent_dir)
            if selected_target_version != self.msvc_arch_str:
                host_dir = os.path.dirname(compiler_parent_dir)
                self.compiler = os.path.join(host_dir, self.msvc_arch_str, 'cl.exe')
                if self.verbose:
                    print('Using alternate compiler "{0}" to match selected target.'.format(self.compiler))

        if self.mode == 'link' or self.mode == 'compile-and-link':
            self.linker = self._find_linker('link') if toolchain_type == 'msvc' else self._find_linker('lld-link', args.tools_dir)
            if not self.linker:
                raise ValueError('Unable to find an appropriate linker.')

        self.compile_env, self.link_env = self._get_visual_studio_environment()

    def _find_linker(self, name, search_paths=[]):
        compiler_dir = os.path.dirname(self.compiler)
        linker_path = find_executable(name, [compiler_dir] + search_paths)
        if linker_path is None:
            raise ValueError('Could not find \'{}\''.format(name))
        return linker_path

    def _get_vc_install_dir(self):
        dir = os.getenv('VCINSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VCINSTALLDIR% {}'.format(dir))
            return dir

        dir = os.getenv('VSINSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VSINSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2019INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2019INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2017INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2017INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')

        dir = os.getenv('VS2015INSTALLDIR', None)
        if dir:
            if self.verbose:
                print('Using %VS2015INSTALLDIR% {}'.format(dir))
            return os.path.join(dir, 'VC')
        return None

    def _get_vctools_version(self):
        ver = os.getenv('VCToolsVersion', None)
        if ver:
            if self.verbose:
                print('Using %VCToolsVersion% {}'.format(ver))
            return ver

        vcinstalldir = self._get_vc_install_dir()
        vcinstalldir = os.path.join(vcinstalldir, 'Tools', 'MSVC')
        subdirs = next(os.walk(vcinstalldir))[1]
        if not subdirs:
            return None

        from distutils.version import StrictVersion
        subdirs.sort(key=lambda x : StrictVersion(x))

        if self.verbose:
            full_path = os.path.join(vcinstalldir, subdirs[-1])
            print('Using VC tools version directory {0} found by directory walk.'.format(full_path))
        return subdirs[-1]

    def _get_vctools_install_dir(self):
        dir = os.getenv('VCToolsInstallDir', None)
        if dir:
            if self.verbose:
                print('Using %VCToolsInstallDir% {}'.format(dir))
            return dir

        vcinstalldir = self._get_vc_install_dir()
        if not vcinstalldir:
            return None
        vctoolsver = self._get_vctools_version()
        if not vctoolsver:
            return None
        result = os.path.join(vcinstalldir, 'Tools', 'MSVC', vctoolsver)
        if not os.path.exists(result):
            return None
        if self.verbose:
            print('Using VC tools install dir {} found by directory walk'.format(result))
        return result

    def _find_windows_sdk_in_registry_view(self, view):
        products_key = None
        roots_key = None
        installed_options_keys = []
        try:
            sam = view | winreg.KEY_READ
            products_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                          r'Software\Microsoft\Windows Kits\Installed Products',
                                          0,
                                          sam)

            # This is the GUID for the desktop component.  If this is present
            # then the components required for the Desktop SDK are installed.
            # If not it will throw an exception.
            winreg.QueryValueEx(products_key, '{5A3D81EC-D870-9ECF-D997-24BDA6644752}')

            roots_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                       r'Software\Microsoft\Windows Kits\Installed Roots',
                                       0,
                                       sam)
            root_dir = winreg.QueryValueEx(roots_key, 'KitsRoot10')
            root_dir = to_string(root_dir[0])
            sdk_versions = []
            index = 0
            while True:
                # Installed SDK versions are stored as sub-keys of the
                # 'Installed Roots' key.  Find all of their names, then sort
                # them by version
                try:
                    ver_key = winreg.EnumKey(roots_key, index)
                    sdk_versions.append(ver_key)
                    index = index + 1
                except WindowsError:
                    break
            if not sdk_versions:
                return (None, None)

            # Windows SDK version numbers consist of 4 dotted components, so we
            # have to use LooseVersion, as StrictVersion supports 3 or fewer.
            from distutils.version import LooseVersion
            sdk_versions.sort(key=lambda x : LooseVersion(x), reverse=True)
            option_value_name = 'OptionId.DesktopCPP' + self.msvc_arch_str
            for v in sdk_versions:
                try:
                    version_subkey = v + r'\Installed Options'
                    key = winreg.OpenKey(roots_key, version_subkey)
                    installed_options_keys.append(key)
                    (value, value_type) = winreg.QueryValueEx(key, option_value_name)
                    if value == 1:
                        # The proper architecture is installed.  Return the
                        # associated paths.
                        if self.verbose:
                            print('Found Installed Windows SDK v{0} at {1}'.format(v, root_dir))
                        return (root_dir, v)
                except:
                    continue
        except:
            return (None, None)
        finally:
            del products_key
            del roots_key
            for k in installed_options_keys:
                del k
        return (None, None)

    def _find_windows_sdk_in_registry(self):
        # This could be a clang-cl cross-compile.  If so, there's no registry
        # so just exit.
        if sys.platform != 'win32':
            return (None, None)
        if self.verbose:
            print('Looking for Windows SDK in 64-bit registry.')
        dir, ver = self._find_windows_sdk_in_registry_view(winreg.KEY_WOW64_64KEY)
        if not dir or not ver:
            if self.verbose:
                print('Looking for Windows SDK in 32-bit registry.')
            dir, ver = self._find_windows_sdk_in_registry_view(winreg.KEY_WOW64_32KEY)

        return (dir, ver)

    def _get_winsdk_dir(self):
        # If a Windows SDK is specified in the environment, use that.  Otherwise
        # try to find one in the Windows registry.
        dir = os.getenv('WindowsSdkDir', None)
        if not dir or not os.path.exists(dir):
            return self._find_windows_sdk_in_registry()
        ver = os.getenv('WindowsSDKLibVersion', None)
        if not ver:
            return self._find_windows_sdk_in_registry()

        ver = ver.rstrip('\\')
        if self.verbose:
            print('Using %WindowsSdkDir% {}'.format(dir))
            print('Using %WindowsSDKLibVersion% {}'.format(ver))
        return (dir, ver)

    def _get_msvc_native_toolchain_dir(self):
        assert self.toolchain_type == 'msvc'
        compiler_dir = os.path.dirname(self.compiler)
        target_dir = os.path.dirname(compiler_dir)
        host_name = os.path.basename(target_dir)
        host_name = host_name[4:].lower()
        return os.path.join(target_dir, host_name)

    def _get_visual_studio_environment(self):
        vctools = self._get_vctools_install_dir()
        winsdk, winsdkver = self._get_winsdk_dir()

        if not vctools and self.verbose:
            print('Unable to find VC tools installation directory.')
        if (not winsdk or not winsdkver) and self.verbose:
            print('Unable to find Windows SDK directory.')

        vcincludes = []
        vclibs = []
        sdkincludes = []
        sdklibs = []
        if vctools is not None:
            includes = [['ATLMFC', 'include'], ['include']]
            libs = [['ATLMFC', 'lib'], ['lib']]
            vcincludes = [os.path.join(vctools, *y) for y in includes]
            vclibs = [os.path.join(vctools, *y) for y in libs]
        if winsdk is not None:
            includes = [['include', winsdkver, 'ucrt'],
                        ['include', winsdkver, 'shared'],
                        ['include', winsdkver, 'um'],
                        ['include', winsdkver, 'winrt'],
                        ['include', winsdkver, 'cppwinrt']]
            libs = [['lib', winsdkver, 'ucrt'],
                    ['lib', winsdkver, 'um']]
            sdkincludes = [os.path.join(winsdk, *y) for y in includes]
            sdklibs = [os.path.join(winsdk, *y) for y in libs]

        includes = vcincludes + sdkincludes
        libs = vclibs + sdklibs
        libs = [os.path.join(x, self.msvc_arch_str) for x in libs]
        compileenv = None
        linkenv = None
        defaultenv = {}
        if sys.platform == 'win32':
            defaultenv = { x : os.environ[x] for x in
                          ['SystemDrive', 'SystemRoot', 'TMP', 'TEMP'] }
            # The directory to mspdbcore.dll needs to be in PATH, but this is
            # always in the native toolchain path, not the cross-toolchain
            # path.  So, for example, if we're using HostX64\x86 then we need
            # to add HostX64\x64 to the path, and if we're using HostX86\x64
            # then we need to add HostX86\x86 to the path.
            if self.toolchain_type == 'msvc':
                defaultenv['PATH'] = self._get_msvc_native_toolchain_dir()

        if includes:
            compileenv = {}
            compileenv['INCLUDE'] = os.pathsep.join(includes)
            compileenv.update(defaultenv)
        if libs:
            linkenv = {}
            linkenv['LIB'] = os.pathsep.join(libs)
            linkenv.update(defaultenv)
        return (compileenv, linkenv)

    def _ilk_file_names(self):
        if self.mode == 'link':
            return []

        return [self._output_name(x, '.ilk') for x in self.inputs]

    def _pdb_file_name(self):
        if self.mode == 'compile':
            return None
        return os.path.splitext(self.output)[0] + '.pdb'

    def _get_compilation_command(self, source, obj):
        args = []

        args.append(self.compiler)
        if self.toolchain_type == 'clang-cl':
            args.append('-m' + self.arch)

        if self.opt == 'none':
            args.append('/Od')
        elif self.opt == 'basic':
            args.append('/O2')
        elif self.opt == 'lto':
            if self.toolchain_type == 'msvc':
                args.append('/GL')
                args.append('/Gw')
            else:
                args.append('-flto=thin')
        if self.nodefaultlib:
            args.append('/GS-')
            args.append('/GR-')
        args.append('/Z7')
        if self.toolchain_type == 'clang-cl':
            args.append('-Xclang')
            args.append('-fkeep-static-consts')
            args.append('-fms-compatibility-version=19')
        args.append('/c')

        args.append('/Fo' + obj)
        if self.toolchain_type == 'clang-cl':
            args.append('--')
        args.append(source)

        return ('compiling', [source], obj,
                self.compile_env,
                args)

    def _get_link_command(self):
        args = []
        args.append(self.linker)
        args.append('/DEBUG:FULL')
        args.append('/INCREMENTAL:NO')
        if self.nodefaultlib:
            args.append('/nodefaultlib')
            args.append('/entry:main')
        args.append('/PDB:' + self._pdb_file_name())
        args.append('/OUT:' + self._exe_file_name())
        args.extend(self._obj_file_names())

        return ('linking', self._obj_file_names(), self._exe_file_name(),
                self.link_env,
                args)

    def build_commands(self):
        commands = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            for input, output in zip(self.inputs, self._obj_file_names()):
                commands.append(self._get_compilation_command(input, output))
        if self.mode == 'link' or self.mode == 'compile-and-link':
            commands.append(self._get_link_command())
        return commands

    def output_files(self):
        outputs = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            outputs.extend(self._ilk_file_names())
            outputs.extend(self._obj_file_names())
        if self.mode == 'link' or self.mode == 'compile-and-link':
            outputs.append(self._pdb_file_name())
            outputs.append(self._exe_file_name())

        return [x for x in outputs if x is not None]

class GccBuilder(Builder):
    def __init__(self, toolchain_type, args):
        Builder.__init__(self, toolchain_type, args, '.o')
        if sys.platform == 'darwin':
            cmd = ['xcrun', '--sdk', args.apple_sdk, '--show-sdk-path']
            self.apple_sdk = subprocess.check_output(cmd).strip().decode('utf-8')

    def _get_compilation_command(self, source, obj):
        args = []

        args.append(self.compiler)
        args.append('-m' + self.arch)

        args.append('-g')
        if self.opt == 'none':
            args.append('-O0')
        elif self.opt == 'basic':
            args.append('-O2')
        elif self.opt == 'lto':
            args.append('-flto=thin')
        if self.nodefaultlib:
            args.append('-nostdinc')
            args.append('-static')
        args.append('-c')

        args.extend(['-o', obj])
        args.append(source)

        if sys.platform == 'darwin':
            args.extend(['-isysroot', self.apple_sdk])

        return ('compiling', [source], obj, None, args)

    def _get_link_command(self):
        args = []
        args.append(self.compiler)
        args.append('-m' + self.arch)
        if self.nodefaultlib:
            args.append('-nostdlib')
            args.append('-static')
            main_symbol = 'main'
            if sys.platform == 'darwin':
                main_symbol = '_main'
            args.append('-Wl,-e,' + main_symbol)
        if sys.platform.startswith('netbsd'):
            for x in self.lib_paths:
                args += ['-L' + x, '-Wl,-rpath,' + x]
        args.extend(['-o', self._exe_file_name()])
        args.extend(self._obj_file_names())

        if sys.platform == 'darwin':
            args.extend(['-isysroot', self.apple_sdk])

        return ('linking', self._obj_file_names(), self._exe_file_name(), None, args)


    def output_files(self):
        outputs = []
        if self.mode == 'compile' or self.mode == 'compile-and-link':
            outputs.extend(self._obj_file_names())
        if self.mode == 'link' or self.mode == 'compile-and-link':
            outputs.append(self._exe_file_name())

        return outputs

def indent(text, spaces):
    def prefixed_lines():
        prefix = ' ' * spaces
        for line in text.splitlines(True):
            yield prefix + line
    return ''.join(prefixed_lines())

def build(commands):
    global args
    for (status, inputs, output, env, child_args) in commands:
        print('\n\n')
        inputs = [os.path.basename(x) for x in inputs]
        output = os.path.basename(output)
        print(status + ' {0} -> {1}'.format('+'.join(inputs), output))

        if args.verbose:
            print('  Command Line: ' + ' '.join(child_args))
            print('  Env:')
            print_environment(env)
        if args.dry:
            continue

        popen = subprocess.Popen(child_args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env,
                                 universal_newlines=True)
        stdout, stderr = popen.communicate()
        res = popen.wait()
        if res == -signal.SIGINT:
            raise KeyboardInterrupt
        print('  STDOUT:')
        print(indent(stdout, 4))
        if res != 0:
            print('  STDERR:')
            print(indent(stderr, 4))
            sys.exit(res)

def clean(files):
    global args
    if not files:
        return
    for o in files:
        file = o if args.verbose else os.path.basename(o)
        print('Cleaning {0}'.format(file))
        try:
            if os.path.exists(o):
                if not args.dry:
                    os.remove(o)
                if args.verbose:
                    print('  The file was successfully cleaned.')
            elif args.verbose:
                print('  The file does not exist.')
        except:
            if args.verbose:
                print('  The file could not be removed.')

def fix_arguments(args):
    if not args.inputs:
        raise ValueError('No input files specified')

    if args.output and args.mode == 'compile' and len(args.inputs) > 1:
        raise ValueError('Cannot specify -o with mode=compile and multiple source files.  Use --outdir instead.')

    if not args.dry:
        args.inputs = [os.path.abspath(x) for x in args.inputs]

    # If user didn't specify the outdir, use the directory of the first input.
    if not args.outdir:
        if args.output:
            args.outdir = os.path.dirname(args.output)
        else:
            args.outdir = os.path.dirname(args.inputs[0])
            args.outdir = os.path.abspath(args.outdir)
        args.outdir = os.path.normpath(args.outdir)

    # If user specified a non-absolute path for the output file, append the
    # output directory to it.
    if args.output:
        if not os.path.isabs(args.output):
            args.output = os.path.join(args.outdir, args.output)
        args.output = os.path.normpath(args.output)

fix_arguments(args)

(toolchain_type, toolchain_path) = find_toolchain(args.compiler, args.tools_dir)
if not toolchain_path or not toolchain_type:
    print('Unable to find toolchain {0}'.format(args.compiler))
    sys.exit(1)

if args.verbose:
    print('Script Arguments:')
    print('  Arch: ' + args.arch)
    print('  Compiler: ' + args.compiler)
    print('  Outdir: ' + args.outdir)
    print('  Output: ' + args.output)
    print('  Nodefaultlib: ' + str(args.nodefaultlib))
    print('  Opt: ' + args.opt)
    print('  Mode: ' + args.mode)
    print('  Clean: ' + str(args.clean))
    print('  Verbose: ' + str(args.verbose))
    print('  Dryrun: ' + str(args.dry))
    print('  Inputs: ' + format_text(args.inputs, 0, 10))
    print('Script Environment:')
    print_environment(os.environ)

args.compiler = toolchain_path
if not os.path.exists(args.compiler) and not args.dry:
    raise ValueError('The toolchain {} does not exist.'.format(args.compiler))

if toolchain_type == 'msvc' or toolchain_type=='clang-cl':
    builder = MsvcBuilder(toolchain_type, args)
else:
    builder = GccBuilder(toolchain_type, args)

if args.clean:
    clean(builder.output_files())

cmds = builder.build_commands()

build(cmds)
