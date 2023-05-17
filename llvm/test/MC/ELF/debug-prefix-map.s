# RUN: rm -rf %t.foo && mkdir -p %t.foo/bar && cd %t.foo
# RUN: cp %s %t.foo/src.s && cp %s %t.foo/bar/src.s

# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=4 src.s -filetype=obj -o nomap.4.o
# RUN: llvm-dwarfdump -v -debug-info -debug-line nomap.4.o | FileCheck --check-prefix=NO_MAP_V4 %s
# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=5 src.s -filetype=obj -o nomap.5.o
# RUN: llvm-dwarfdump -v -debug-info -debug-line nomap.5.o | FileCheck --check-prefix=NO_MAP_V5 %s

# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=4 src.s -filetype=obj -o map.4.o -fdebug-prefix-map=%t.foo=src_root
# RUN: llvm-dwarfdump -v -debug-info -debug-line map.4.o | FileCheck --check-prefix=MAP_V4 %s
# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=5 src.s -filetype=obj -o map.5.o -fdebug-prefix-map=%t.foo=src_root
# RUN: llvm-dwarfdump -v -debug-info -debug-line map.5.o | FileCheck --check-prefix=MAP_V5 %s

# RUN: cd %t.foo/bar
# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=4 %t.foo%{fs-sep}bar%{fs-sep}src.s -filetype=obj -o mapabs.4.o -fdebug-prefix-map=%t.foo=/broken_root -fdebug-prefix-map=%t.foo%{fs-sep}bar=/src_root
# RUN: llvm-dwarfdump -v -debug-info -debug-line mapabs.4.o | FileCheck --check-prefix=MAPABS_V4 %s
# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=5 %t.foo%{fs-sep}bar%{fs-sep}src.s -filetype=obj -o mapabs.5.o -fdebug-prefix-map=%t.foo=/broken_root -fdebug-prefix-map=%t.foo%{fs-sep}bar=/src_root
# RUN: llvm-dwarfdump -v -debug-info -debug-line mapabs.5.o | FileCheck --check-prefix=MAPABS_V5 %s

## The last -fdebug-prefix-map= wins even if the prefix is shorter.
# RUN: llvm-mc -triple=x86_64 -g -dwarf-version=5 %t.foo%{fs-sep}bar%{fs-sep}src.s -filetype=obj -o mapabs.5.o -fdebug-prefix-map=%t.foo%{fs-sep}bar=/broken_root -fdebug-prefix-map=%t.foo=/src_root
# RUN: llvm-dwarfdump -v -debug-info -debug-line mapabs.5.o | FileCheck --check-prefix=MAPABS_V5_A %s

f:
  nop

# NO_MAP_V4:      DW_AT_comp_dir [DW_FORM_string] ("{{.*}}.foo")
# NO_MAP_V4:      file_names[  1]:
# NO_MAP_V4-NEXT:   name: "src.s"

# NO_MAP_V5:      DW_AT_comp_dir [DW_FORM_string] ("{{.*}}.foo")
# NO_MAP_V5:      include_directories[  0] =  .debug_line_str[0x00000000] = "{{.*}}.foo"

# MAP_V4:      DW_AT_name [DW_FORM_string] ("src.s")
# MAP_V4:      DW_AT_comp_dir [DW_FORM_string] ("src_root")
# MAP_V4:      DW_AT_decl_file [DW_FORM_data4] ("src_root{{(/|\\)+}}src.s")
# MAP_V4-NOT:  .foo

# MAP_V5:      DW_AT_name [DW_FORM_string] ("src.s")
# MAP_V5:      DW_AT_comp_dir [DW_FORM_string] ("src_root")
# MAP_V5:      DW_AT_decl_file [DW_FORM_data4] ("src_root{{(/|\\)+}}src.s")
# MAP_V5:      include_directories[ 0] = .debug_line_str[0x00000000] = "src_root"

# MAPABS_V4:      DW_AT_name [DW_FORM_string] ("src.s")
# MAPABS_V4:      DW_AT_comp_dir [DW_FORM_string] ("{{(/|\\)+}}src_root")
# MAPABS_V4:      DW_AT_decl_file [DW_FORM_data4] ("/src_root{{(/|\\)+}}src.s")
# MAPABS_V4-NOT:  .foo

# MAPABS_V5:      DW_AT_name [DW_FORM_string] ("src.s")
# MAPABS_V5:      DW_AT_comp_dir [DW_FORM_string] ("{{(/|\\)+}}src_root")
# MAPABS_V5:      DW_AT_decl_file [DW_FORM_data4] ("/src_root{{(/|\\)+}}src.s")
# MAPABS_V5:      include_directories[ 0] = .debug_line_str[0x00000000] = "/src_root"

# MAPABS_V5_A:    DW_AT_comp_dir [DW_FORM_string] ("{{(/|\\)+}}src_root{{(/|\\)+}}bar")
# MAPABS_V5_A:    DW_AT_decl_file [DW_FORM_data4] ("/src_root{{(/|\\)+}}bar{{(/|\\)+}}src.s")
