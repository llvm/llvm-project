## Test the functionality of extract_ir_lib

import os.path
import sys

from mlgo.corpus import extract_ir_lib

## Test that we can convert a compilation database with a single compilation
## command in it.

# RUN: %python %s test_one_conversion | FileCheck %s --check-prefix CHECK-ONE-CONVERSION

# CHECK-ONE-CONVERSION: /output/directory/lib/bar.o
# CHECK-ONE-CONVERSION: lib/bar.o
# CHECK-ONE-CONVERSION: /corpus/destination/path/lib/bar.o.cmd
# CHECK-ONE-CONVERSION: /corpus/destination/path/lib/bar.o.bc
# CHECK-ONE-CONVERSION: /corpus/destination/path/lib/bar.o.thinlto.bc

def test_one_conversion():
    obj = extract_ir_lib.convert_compile_command_to_objectfile(
        {
            "directory": "/output/directory",
            "command": "-cc1 -c /some/path/lib/foo/bar.cc -o lib/bar.o",
            "file": "/some/path/lib/foo/bar.cc",
        },
        "/corpus/destination/path",
    )
    print(obj.input_obj())
    print(obj.relative_output_path())
    print(obj.cmd_file())
    print(obj.bc_file())
    print(obj.thinlto_index_file())

## Test that we can convert an arguments style compilation database

# RUN: %python %s test_one_conversion_arguments_style | FileCheck %s --check-prefix CHECK-ARGUMENTS-STYLE

# CHECK-ARGUMENTS-STYLE: /output/directory/lib/bar.o
# CHECK-ARGUMENTS-STYLE: lib/bar.o
# CHECK-ARGUMENTS-STYLE: /corpus/destination/path/lib/bar.o.cmd
# CHECK-ARGUMENTS-STYLE: /corpus/destination/path/lib/bar.o.bc
# CHECK-ARGUMENTS-STYLE: /corpus/destination/path/lib/bar.o.thinlto.bc

def test_one_conversion_arguments_style():
    obj = extract_ir_lib.convert_compile_command_to_objectfile(
        {
            "directory": "/output/directory",
            "arguments": [
                "-cc1",
                "-c",
                "/some/path/lib/foo/bar.cc",
                "-o",
                "lib/bar.o",
            ],
            "file": "/some/path/lib/foo/bar.cc",
        },
        "/corpus/destination/path",
    )
    print(obj.input_obj())
    print(obj.relative_output_path())
    print(obj.cmd_file())
    print(obj.bc_file())
    print(obj.thinlto_index_file())

## Test that converting multiple files works as well

# RUN: %python %s test_multiple_conversion | FileCheck %s --check-prefix CHECK-MULTIPLE-CONVERSION

# CHECK-MULTIPLE-CONVERSION: /output/directory/lib/bar.o
# CHECK-MULTIPLE-CONVERSION: lib/bar.o
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/bar.o.cmd
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/bar.o.bc
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/bar.o.thinlto.bc
# CHECK-MULTIPLE-CONVERSION: /output/directory/lib/other/baz.o
# CHECK-MULTIPLE-CONVERSION: lib/other/baz.o
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/other/baz.o.cmd
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/other/baz.o.bc
# CHECK-MULTIPLE-CONVERSION: /corpus/destination/path/lib/other/baz.o.thinlto.bc

def test_multiple_conversion():
    res = extract_ir_lib.load_from_compile_commands(
        [
            {
                "directory": "/output/directory",
                "command": "-cc1 -c /some/path/lib/foo/bar.cc -o lib/bar.o",
                "file": "/some/path/lib/foo/bar.cc",
            },
            {
                "directory": "/output/directory",
                "command": "-cc1 -c /some/path/lib/foo/baz.cc -o lib/other/baz.o",
                "file": "/some/path/lib/foo/baz.cc",
            },
        ],
        "/corpus/destination/path",
    )
    res = list(res)
    print(res[0].input_obj())
    print(res[0].relative_output_path())
    print(res[0].cmd_file())
    print(res[0].bc_file())
    print(res[0].thinlto_index_file())

    print(res[1].input_obj(), "/output/directory/lib/other/baz.o")
    print(res[1].relative_output_path(), "lib/other/baz.o")
    print(res[1].cmd_file())
    print(res[1].bc_file())
    print(res[1].thinlto_index_file())

## Test that we generate the correct objcopy commands for extracting commands

# RUN: %python %s test_command_extraction | FileCheck %s --check-prefix CHECK-COMMAND-EXTRACT

# CHECK-COMMAND-EXTRACT: /bin/llvm_objcopy_path
# CHECK-COMMAND-EXTRACT: --dump-section=.llvmcmd=/where/corpus/goes/lib/obj_file.o.cmd
# CHECK-COMMAND-EXTRACT: /foo/bar/lib/obj_file.o
# CHECK-COMMAND-EXTRACT: /dev/null
# CHECK-COMMAND-EXTRACT: /bin/llvm_objcopy_path
# CHECK-COMMAND-EXTRACT: --dump-section=.llvmbc=/where/corpus/goes/lib/obj_file.o.bc
# CHECK-COMMAND-EXTRACT: /foo/bar/lib/obj_file.o
# CHECK-COMMAND-EXTRACT: /dev/null

def test_command_extraction():
    obj = extract_ir_lib.TrainingIRExtractor(
        obj_relative_path="lib/obj_file.o",
        output_base_dir="/where/corpus/goes",
        obj_base_dir="/foo/bar",
    )
    extraction_cmd1 = obj._get_extraction_cmd_command("/bin/llvm_objcopy_path", ".llvmcmd")
    for part in extraction_cmd1:
        print(part)

    extraction_cmd2 = obj._get_extraction_bc_command("/bin/llvm_objcopy_path", ".llvmbc")
    for part in extraction_cmd2:
        print(part)

## Test that we generate the correct extraction commands without specifying
## an output base directory.

# RUN: %python %s test_command_extraction_no_basedir | FileCheck %s --check-prefix CHECK-COMMAND-EXTRACT-NOBASEDIR

# CHECK-COMMAND-EXTRACT-NOBASEDIR: /bin/llvm_objcopy_path
# CHECK-COMMAND-EXTRACT-NOBASEDIR: --dump-section=.llvmcmd=/where/corpus/goes/lib/obj_file.o.cmd
# CHECK-COMMAND-EXTRACT-NOBASEDIR: lib/obj_file.o
# CHECK-COMMAND-EXTRACT-NOBASEDIR: /dev/null
# CHECK-COMMAND-EXTRACT-NOBASEDIR: /bin/llvm_objcopy_path
# CHECK-COMMAND-EXTRACT-NOBASEDIR: --dump-section=.llvmbc=/where/corpus/goes/lib/obj_file.o.bc
# CHECK-COMMAND-EXTRACT-NOBASEDIR: lib/obj_file.o
# CHECK-COMMAND-EXTRACT-NOBASEDIR: /dev/null

def test_command_extraction_no_basedir():
    obj = extract_ir_lib.TrainingIRExtractor("lib/obj_file.o", "/where/corpus/goes")
    extraction_cmd1 = obj._get_extraction_cmd_command("/bin/llvm_objcopy_path", ".llvmcmd")
    for part in extraction_cmd1:
        print(part)

    extraction_cmd2 = obj._get_extraction_bc_command("/bin/llvm_objcopy_path", ".llvmbc")
    for part in extraction_cmd2:
        print(part)

## Test that we can extract a corpus from lld parameters

# RUN: %python %s test_lld_params | FileCheck %s --check-prefix CHECK-LLD-PARAMS

# CHECK-LLD-PARAMS: /some/path/lib/obj1.o
# CHECK-LLD-PARAMS: lib/obj1.o
# CHECK-LLD-PARAMS: /tmp/out/lib/obj1.o.cmd
# CHECK-LLD-PARAMS: /tmp/out/lib/obj1.o.thinlto.bc
# CHECK-LLD-PARMAS: /some/path/lib/dir/obj2.o

def test_lld_params():
    lld_opts = [
        "-o",
        "output/dir/exe",
        "lib/obj1.o",
        "somelib.a",
        "-W,blah",
        "lib/dir/obj2.o",
    ]
    obj = extract_ir_lib.load_from_lld_params(lld_opts, "/some/path", "/tmp/out")
    print(obj[0].input_obj())
    print(obj[0].relative_output_path())
    print(obj[0].cmd_file())
    print(obj[0].thinlto_index_file())
    print(obj[1].input_obj())

## Test that we can load a corpus from a directory containing object files

# RUN: rm -rf %t.dir && mkdir %t.dir
# RUN: mkdir %t.dir/subdir
# RUN: touch %t.dir/subdir/test1.o
# RUN: touch %t.dir/subdir/test2.o
# RUN: %python %s test_load_from_directory %t.dir /output | FileCheck %s --check-prefix CHECK-LOAD-DIR

# CHECK-LOAD-DIR: subdir/test1.o
# CHECK-LOAD-DIR: True
# CHECK-LOAD-DIR /output

def test_load_from_directory(tempdir, outdir):
    objs = extract_ir_lib.load_from_directory(tempdir, outdir)
    for index, obj in enumerate(sorted(objs, key=lambda x: x._obj_relative_path)):
        print(obj._obj_relative_path, f"subdir/test{index + 1:d}.o")
        # Explicitly check for equality here as we can not check within
        # FileCheck the exact value as lit substitutions do not work in
        # FileCheck lines.
        print(obj._obj_base_dir == tempdir)
        print(obj._output_base_dir)

def test_lld_thinlto_discovery(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path="1.3.import.bc")
    tempdir.create_file(file_path="2.3.import.bc")
    tempdir.create_file(file_path="3.3.import.bc")
    tempdir.create_file(file_path="1.thinlto.bc")
    tempdir.create_file(file_path="2.thinlto.bc")
    tempdir.create_file(file_path="3.thinlto.bc")
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(tempdir.full_path, outdir.full_path)
    self.assertLen(obj, 3)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
        print(o._obj_relative_path, f"{i + 1:d}")
        print(o._obj_base_dir, tempdir.full_path)
        print(o._output_base_dir, outdir.full_path)

def test_lld_thinlto_discovery_nested(self):
    outer = self.create_tempdir()
    tempdir = outer.mkdir(dir_path="nest")
    tempdir.create_file(file_path="1.3.import.bc")
    tempdir.create_file(file_path="2.3.import.bc")
    tempdir.create_file(file_path="3.3.import.bc")
    tempdir.create_file(file_path="1.thinlto.bc")
    tempdir.create_file(file_path="2.thinlto.bc")
    tempdir.create_file(file_path="3.thinlto.bc")
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(outer.full_path, outdir.full_path)
    self.assertLen(obj, 3)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
        print(o._obj_relative_path, f"nest/{i + 1:d}")
        print(o._obj_base_dir, outer.full_path)
        print(o._output_base_dir, outdir.full_path)

def test_lld_thinlto_extraction(self):
    outer = self.create_tempdir()
    tempdir = outer.mkdir(dir_path="nest")
    tempdir.create_file(file_path="1.3.import.bc")
    tempdir.create_file(file_path="2.3.import.bc")
    tempdir.create_file(file_path="3.3.import.bc")
    tempdir.create_file(file_path="1.thinlto.bc")
    tempdir.create_file(file_path="2.thinlto.bc")
    tempdir.create_file(file_path="3.thinlto.bc")
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(outer.full_path, outdir.full_path)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
        mod_path = o.extract(thinlto_build="local")
        print(mod_path, f"nest/{i + 1:d}")
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, "nest/1.bc")))
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, "nest/2.bc")))
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, "nest/3.bc")))
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, "nest/1.thinlto.bc"))
    )
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, "nest/2.thinlto.bc"))
    )
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, "nest/3.thinlto.bc"))
    )

def test_filtering(self):
    cmdline = "-cc1\0x/y/foobar.cpp\0-Oz\0-Ifoo\0-o\0bin/out.o"
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, None))
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, ".*"))
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, "^-Oz$"))
    self.assertFalse(extract_ir_lib.should_include_module(cmdline, "^-O3$"))

def test_thinlto_index_extractor(self):
    cmdline = (
        "-cc1\0x/y/foobar.cpp\0-Oz\0-Ifoo\0-o\0bin/"
        "out.o\0-fthinlto-index=foo/bar.thinlto.bc"
    )
    print(
        extract_ir_lib.get_thinlto_index(cmdline, "/the/base/dir"),
        "/the/base/dir/foo/bar.thinlto.bc",
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        globals()[sys.argv[1]](*sys.argv[2:])
    else:
        globals()[sys.argv[1]]()
