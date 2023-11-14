# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for compiler_opt.tools.extract_ir."""

# pylint: disable=protected-access
import os.path

from absl.testing import absltest

from compiler_opt.tools import extract_ir_lib


class ExtractIrTest(absltest.TestCase):

  def test_one_conversion(self):
    obj = extract_ir_lib.convert_compile_command_to_objectfile(
        {
            'directory': '/output/directory',
            'command': '-cc1 -c /some/path/lib/foo/bar.cc -o lib/bar.o',
            'file': '/some/path/lib/foo/bar.cc'
        }, '/corpus/destination/path')
    self.assertIsNotNone(obj)
    # pytype: disable=attribute-error
    # Pytype complains about obj being None
    self.assertEqual(obj.input_obj(), '/output/directory/lib/bar.o')
    self.assertEqual(obj.relative_output_path(), 'lib/bar.o')
    self.assertEqual(obj.cmd_file(), '/corpus/destination/path/lib/bar.o.cmd')
    self.assertEqual(obj.bc_file(), '/corpus/destination/path/lib/bar.o.bc')
    self.assertEqual(obj.thinlto_index_file(),
                     '/corpus/destination/path/lib/bar.o.thinlto.bc')
    # pytype: enable=attribute-error

  def test_one_conversion_arguments_style(self):
    obj = extract_ir_lib.convert_compile_command_to_objectfile(
        {
            'directory': '/output/directory',
            'arguments':
                ['-cc1', '-c', '/some/path/lib/foo/bar.cc', '-o', 'lib/bar.o'],
            'file': '/some/path/lib/foo/bar.cc'
        }, '/corpus/destination/path')
    self.assertIsNotNone(obj)
    # pytype: disable=attribute-error
    # Pytype complains about obj being None
    self.assertEqual(obj.input_obj(), '/output/directory/lib/bar.o')
    self.assertEqual(obj.relative_output_path(), 'lib/bar.o')
    self.assertEqual(obj.cmd_file(), '/corpus/destination/path/lib/bar.o.cmd')
    self.assertEqual(obj.bc_file(), '/corpus/destination/path/lib/bar.o.bc')
    self.assertEqual(obj.thinlto_index_file(),
                     '/corpus/destination/path/lib/bar.o.thinlto.bc')
    # pytype: enable=attribute-error

  def test_arr_conversion(self):
    res = extract_ir_lib.load_from_compile_commands([{
        'directory': '/output/directory',
        'command': '-cc1 -c /some/path/lib/foo/bar.cc -o lib/bar.o',
        'file': '/some/path/lib/foo/bar.cc'
    }, {
        'directory': '/output/directory',
        'command': '-cc1 -c /some/path/lib/foo/baz.cc -o lib/other/baz.o',
        'file': '/some/path/lib/foo/baz.cc'
    }], '/corpus/destination/path')
    res = list(res)
    self.assertLen(res, 2)
    self.assertEqual(res[0].input_obj(), '/output/directory/lib/bar.o')
    self.assertEqual(res[0].relative_output_path(), 'lib/bar.o')
    self.assertEqual(res[0].cmd_file(),
                     '/corpus/destination/path/lib/bar.o.cmd')
    self.assertEqual(res[0].bc_file(), '/corpus/destination/path/lib/bar.o.bc')
    self.assertEqual(res[0].thinlto_index_file(),
                     '/corpus/destination/path/lib/bar.o.thinlto.bc')

    self.assertEqual(res[1].input_obj(), '/output/directory/lib/other/baz.o')
    self.assertEqual(res[1].relative_output_path(), 'lib/other/baz.o')
    self.assertEqual(res[1].cmd_file(),
                     '/corpus/destination/path/lib/other/baz.o.cmd')
    self.assertEqual(res[1].bc_file(),
                     '/corpus/destination/path/lib/other/baz.o.bc')
    self.assertEqual(res[1].thinlto_index_file(),
                     '/corpus/destination/path/lib/other/baz.o.thinlto.bc')

  def test_command_extraction(self):
    obj = extract_ir_lib.TrainingIRExtractor(
        obj_relative_path='lib/obj_file.o',
        output_base_dir='/where/corpus/goes',
        obj_base_dir='/foo/bar')
    self.assertEqual(
        obj._get_extraction_cmd_command('/bin/llvm_objcopy_path', '.llvmcmd'), [
            '/bin/llvm_objcopy_path',
            '--dump-section=.llvmcmd=/where/corpus/goes/lib/obj_file.o.cmd',
            '/foo/bar/lib/obj_file.o', '/dev/null'
        ])
    self.assertEqual(
        obj._get_extraction_bc_command('/bin/llvm_objcopy_path', '.llvmbc'), [
            '/bin/llvm_objcopy_path',
            '--dump-section=.llvmbc=/where/corpus/goes/lib/obj_file.o.bc',
            '/foo/bar/lib/obj_file.o', '/dev/null'
        ])

  def test_command_extraction_no_basedir(self):
    obj = extract_ir_lib.TrainingIRExtractor('lib/obj_file.o',
                                             '/where/corpus/goes')
    self.assertEqual(
        obj._get_extraction_cmd_command('/bin/llvm_objcopy_path', '.llvmcmd'), [
            '/bin/llvm_objcopy_path',
            '--dump-section=.llvmcmd=/where/corpus/goes/lib/obj_file.o.cmd',
            'lib/obj_file.o', '/dev/null'
        ])
    self.assertEqual(
        obj._get_extraction_bc_command('/bin/llvm_objcopy_path', '.llvmbc'), [
            '/bin/llvm_objcopy_path',
            '--dump-section=.llvmbc=/where/corpus/goes/lib/obj_file.o.bc',
            'lib/obj_file.o', '/dev/null'
        ])

  def test_lld_params(self):
    lld_opts = [
        '-o', 'output/dir/exe', 'lib/obj1.o', 'somelib.a', '-W,blah',
        'lib/dir/obj2.o'
    ]
    obj = extract_ir_lib.load_from_lld_params(lld_opts, '/some/path',
                                              '/tmp/out')
    self.assertLen(obj, 2)
    self.assertEqual(obj[0].input_obj(), '/some/path/lib/obj1.o')
    self.assertEqual(obj[0].relative_output_path(), 'lib/obj1.o')
    self.assertEqual(obj[0].cmd_file(), '/tmp/out/lib/obj1.o.cmd')
    self.assertEqual(obj[0].thinlto_index_file(),
                     '/tmp/out/lib/obj1.o.thinlto.bc')
    self.assertEqual(obj[1].input_obj(), '/some/path/lib/dir/obj2.o')

  def test_load_from_directory(self):
    tempdir = self.create_tempdir()
    subdir = tempdir.mkdir(dir_path='subdir')
    subdir.create_file(file_path='test1.o')
    subdir.create_file(file_path='test2.o')
    outdir = self.create_tempdir()
    objs = extract_ir_lib.load_from_directory(tempdir.full_path,
                                              outdir.full_path)
    self.assertLen(objs, 2)
    for index, obj in enumerate(
        sorted(objs, key=lambda x: x._obj_relative_path)):
      self.assertEqual(obj._obj_relative_path, f'subdir/test{index + 1:d}.o')
      self.assertEqual(obj._obj_base_dir, tempdir.full_path)
      self.assertEqual(obj._output_base_dir, outdir.full_path)

  def test_lld_thinlto_discovery(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='1.3.import.bc')
    tempdir.create_file(file_path='2.3.import.bc')
    tempdir.create_file(file_path='3.3.import.bc')
    tempdir.create_file(file_path='1.thinlto.bc')
    tempdir.create_file(file_path='2.thinlto.bc')
    tempdir.create_file(file_path='3.thinlto.bc')
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(tempdir.full_path,
                                              outdir.full_path)
    self.assertLen(obj, 3)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
      self.assertEqual(o._obj_relative_path, f'{i + 1:d}')
      self.assertEqual(o._obj_base_dir, tempdir.full_path)
      self.assertEqual(o._output_base_dir, outdir.full_path)

  def test_lld_thinlto_discovery_nested(self):
    outer = self.create_tempdir()
    tempdir = outer.mkdir(dir_path='nest')
    tempdir.create_file(file_path='1.3.import.bc')
    tempdir.create_file(file_path='2.3.import.bc')
    tempdir.create_file(file_path='3.3.import.bc')
    tempdir.create_file(file_path='1.thinlto.bc')
    tempdir.create_file(file_path='2.thinlto.bc')
    tempdir.create_file(file_path='3.thinlto.bc')
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(outer.full_path, outdir.full_path)
    self.assertLen(obj, 3)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
      self.assertEqual(o._obj_relative_path, f'nest/{i + 1:d}')
      self.assertEqual(o._obj_base_dir, outer.full_path)
      self.assertEqual(o._output_base_dir, outdir.full_path)

  def test_lld_thinlto_extraction(self):
    outer = self.create_tempdir()
    tempdir = outer.mkdir(dir_path='nest')
    tempdir.create_file(file_path='1.3.import.bc')
    tempdir.create_file(file_path='2.3.import.bc')
    tempdir.create_file(file_path='3.3.import.bc')
    tempdir.create_file(file_path='1.thinlto.bc')
    tempdir.create_file(file_path='2.thinlto.bc')
    tempdir.create_file(file_path='3.thinlto.bc')
    outdir = self.create_tempdir()
    obj = extract_ir_lib.load_for_lld_thinlto(outer.full_path, outdir.full_path)
    for i, o in enumerate(sorted(obj, key=lambda x: x._obj_relative_path)):
      mod_path = o.extract(thinlto_build='local')
      self.assertEqual(mod_path, f'nest/{i + 1:d}')
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, 'nest/1.bc')))
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, 'nest/2.bc')))
    self.assertTrue(os.path.exists(os.path.join(outdir.full_path, 'nest/3.bc')))
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, 'nest/1.thinlto.bc')))
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, 'nest/2.thinlto.bc')))
    self.assertTrue(
        os.path.exists(os.path.join(outdir.full_path, 'nest/3.thinlto.bc')))

  def test_filtering(self):
    cmdline = '-cc1\0x/y/foobar.cpp\0-Oz\0-Ifoo\0-o\0bin/out.o'
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, None))
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, '.*'))
    self.assertTrue(extract_ir_lib.should_include_module(cmdline, '^-Oz$'))
    self.assertFalse(extract_ir_lib.should_include_module(cmdline, '^-O3$'))

  def test_thinlto_index_extractor(self):
    cmdline = ('-cc1\0x/y/foobar.cpp\0-Oz\0-Ifoo\0-o\0bin/'
               'out.o\0-fthinlto-index=foo/bar.thinlto.bc')
    self.assertEqual(
        extract_ir_lib.get_thinlto_index(cmdline, '/the/base/dir'),
        '/the/base/dir/foo/bar.thinlto.bc')


if __name__ == '__main__':
  absltest.main()
