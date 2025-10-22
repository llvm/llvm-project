#!D:/Python3/python.exe
# -*- coding: utf-8 -*-

import os
import sys

config_map = {}

def map_config(source_dir, site_config):
    global config_map
    source_dir = os.path.abspath(source_dir)
    source_dir = os.path.normcase(source_dir)
    site_config = os.path.normpath(site_config)
    config_map[source_dir] = site_config

# Set up some builtin parameters, so that by default the LLVM test suite
# configuration file knows how to find the object tree.
builtin_parameters = { 'build_mode' : '.' }

# Allow generated file to be relocatable.
import os
import platform
def path(p):
    if not p: return ''
    # Follows lit.util.abs_path_preserve_drive, which cannot be imported here.
    if platform.system() == 'Windows':
        return os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    else:
        return os.path.realpath(os.path.join(os.path.dirname(__file__), p))


map_config(path(r'..\..\..\..\utils\mlgo-utils\tests\lit.cfg'), path(r'..\utils\mlgo-utils\lit.site.cfg'))
map_config(path(r'..\..\..\..\utils\lit\tests\lit.cfg'), path(r'..\utils\lit\lit.site.cfg'))
map_config(path(r'..\..\..\..\test\lit.cfg.py'), path(r'..\test\lit.site.cfg.py'))
map_config(path(r'..\..\..\..\test\Unit\lit.cfg.py'), path(r'..\test\Unit\lit.site.cfg.py'))

builtin_parameters['config_map'] = config_map

# Make sure we can find the lit package.
llvm_source_root = path(r'..\..\..\..')
sys.path.insert(0, os.path.join(llvm_source_root, 'utils', 'lit'))

if __name__=='__main__':
    from lit.main import main
    main(builtin_parameters)
