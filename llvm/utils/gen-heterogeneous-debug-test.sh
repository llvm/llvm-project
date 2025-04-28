#!/bin/bash

# Script to generate llvm/test/CodeGen/X86/heterogeneous-debug.test

# shellcheck disable=SC2154
set -u

# This is independent of the test we are in, and is not reset in reset_per_test_state
idx=0
inc_idx() { ((idx+=1)); }

# Every other counter/accumulator is per-test and gets reset and the start of a new one
reset_per_test_state() {
  declare -g ir_funcs='' ir_metadata='' mir_funcs='' di_version='' mdid=0
  declare_mdid unit
  declare_mdid file
  declare_mdid dwarf_version
  declare_mdid info_version
}

declare_mdid() {
  ((mdid+=1))
  declare -g "$1=$mdid"
}
cat_generic() { declare -g "$1=${!1}$(cat)"$'\n'; }
cat_ir_funcs() { cat_generic ir_funcs; }
cat_ir_metadata() { cat_generic ir_metadata; }
cat_mir_funcs() { cat_generic mir_funcs; }

print_ir_module() {
cat <<EOF
source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$ir_funcs
declare void @Esc(ptr)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare void @llvm.dbg.def(metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!$unit}
!llvm.module.flags = !{!$dwarf_version, !$info_version}

!$unit = distinct !DICompileUnit(language: DW_LANG_C11, file: !$file, producer: "clang", emissionKind: FullDebug)
!$file = !DIFile(filename: "<stdin>", directory: ".")
!$dwarf_version = !{i32 7, !"Dwarf Version", i32 5}
!$info_version = !{i32 2, !"Debug Info Version", i32 $di_version}
$ir_metadata
EOF
}

# Some read-only helper variables
bit_size_to_byte_size() { printf '%d\n' "$((($1 + 8 - 1) / 8))"; }
readonly scalar_tys=(i1 i4 i8 i16 i17 i32 i64 i128 half bfloat float double fp128)
readonly scalar_ty_bit_sizes=(1 4 8 16 17 32 64 128 16 16 32 64 128)
readonly scalar_ty_byte_sizes=($(for sz in ${scalar_ty_bit_sizes[@]}; do
  bit_size_to_byte_size $sz
done))
readonly scalar_ty_pow2_byte_sizes=($(for sz in ${scalar_ty_bit_sizes[@]}; do
  next_pow2=1
  while [[ $sz -gt $next_pow2 ]]; do
    next_pow2=$(($next_pow2 * 2))
  done
  bit_size_to_byte_size $next_pow2
done))
readonly scalar_ty_bit_masks=($(for sz in ${scalar_ty_bit_sizes[@]}; do
  if (($sz % 8)); then
    printf '%d\n' "$(((1 << $sz) - 1))"
  else
    printf '0\n'
  fi
done))

# Test generation functions

declare_one_var_metadata() {
declare_mdid sub
declare_mdid sub_type
declare_mdid sub_type_types
declare_mdid ret
declare_mdid var
declare_mdid var_type
declare_mdid loc
# FIXME: is the size field never considered? it seems to be irrelevant what it
# is set to as far as the expression is concerned
cat_ir_metadata <<EOF
!$sub = distinct !DISubprogram(name: "Fun$idx", scope: !$file, file: !$file, line: 1, type: !$sub_type, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !$unit, retainedNodes: !$ret)
!$sub_type = !DISubroutineType(types: !$sub_type_types)
!$sub_type_types = !{null}
!$ret = !{}
!$var = !DILocalVariable(name: "Var$idx", scope: !$sub, file: !$file, line: 1, type: !$var_type)
!$var_type = !DIBasicType(name: "int", size: 42, encoding: DW_ATE_signed)
!$loc = !DILocation(scope: !$sub)
EOF
}

add_checks_generic() {
local comment_char="$1"; shift
local cat_callback="$1"; shift
local prefix="$comment_char CHECK-NEXT: "
local checks=''
while (($#)); do
  local check="$1"; shift
  checks+="$prefix$check"$'\n'
done
"$cat_callback" <<EOF
$comment_char CHECK: DW_TAG_variable
$checks$comment_char CHECK-NEXT: DW_AT_name ("Var$idx")
EOF
}
add_checks_ir() { add_checks_generic ';' cat_ir_funcs "$@"; }
add_checks_mir() { add_checks_generic '#' cat_mir_funcs "$@"; }

gencase_ir_one_alloca() {
local type="$1"; shift
local expr="$1"; shift
declare_one_var_metadata
cat_ir_funcs <<EOF
define dso_local void @Fun$idx() #0 !dbg !$sub {
entry:
  %Var$idx = alloca $type
  ; DIExpression($expr)
  call void @llvm.dbg.declare(metadata ptr %Var$idx, metadata !$var, metadata !DIExpression($expr)), !dbg !$loc
  call void @Esc(ptr %Var$idx), !dbg !$loc
  ret void, !dbg !$loc
}
EOF
inc_idx
}

gencase_mir_one_alloca() {
local scalar_ty_idx="$1"; shift
local expr="$1"; shift
local type="${scalar_tys[$scalar_ty_idx]}"
local size="${scalar_ty_byte_sizes[$scalar_ty_idx]}"
local align="${scalar_ty_pow2_byte_sizes[$scalar_ty_idx]}"
declare_one_var_metadata
cat_ir_funcs <<EOF
define dso_local void @Fun$idx() #0 !dbg !$sub {
entry:
  %Var$idx = alloca $type
  ; DIExpression($expr)
  call void @llvm.dbg.declare(metadata ptr %Var$idx, metadata !$var, metadata !DIExpression($expr)), !dbg !$loc
  call void @Esc(ptr %Var$idx), !dbg !$loc
  ret void, !dbg !$loc
}
EOF
cat_mir_funcs <<EOF
---
name:            Fun$idx
tracksRegLiveness: true
registers:
  - { id: 0, class: gr64, preferred-register: '' }
stack:
  - { id: 0, name: Var$idx, type: default, offset: 0, size: $size, alignment: $align,
      debug-info-variable: '!$var', debug-info-expression: '!DIExpression($expr)',
      debug-info-location: '!$loc' }
body:             |
  bb.0.entry:
    RET64 debug-location !$loc
...
EOF
inc_idx
}

gencase_mir_no_alloca_dbg_values() {
declare_one_var_metadata
indent='    '
dbg_values=
while (($#)); do
  local reg="$1"; shift
  local indirect="$1"; shift
  local expr="$1"; shift
  dbg_values+="${indent}; !DIExpression($expr)"$'\n'
  dbg_values+="${indent}DBG_VALUE $reg, $indirect, !$var, !DIExpression($expr), debug-location !$loc"$'\n'
done
cat_ir_funcs <<EOF
define dso_local void @Fun$idx() #0 !dbg !$sub {
entry:
  ret void, !dbg !$loc
}
EOF
cat_mir_funcs <<EOF
---
name:            Fun$idx
tracksRegLiveness: true
registers:
  - { id: 0, class: gr64, preferred-register: '' }
stack: []
body:             |
  bb.0.entry:
$dbg_values
    RET64 debug-location !$loc
...
EOF
inc_idx
}

gencase_heterogeneous_mir_one_alloca() {
local scalar_ty_idx="$1"; shift
local expr="$1"; shift
local type="${scalar_tys[$scalar_ty_idx]}"
local size="${scalar_ty_byte_sizes[$scalar_ty_idx]}"
local align="${scalar_ty_pow2_byte_sizes[$scalar_ty_idx]}"
declare_one_var_metadata
declare_mdid lifetime
cat_ir_metadata <<EOF
!$lifetime = distinct !DILifetime(object: !$var, location: !DIExpr($expr))
EOF
cat_ir_funcs <<EOF
define dso_local void @Fun$idx() #0 !dbg !$sub {
entry:
  %Var$idx = alloca $type
  call void @llvm.dbg.def(metadata !$lifetime, metadata ptr %Var$idx), !dbg !$loc
  call void @Esc(ptr %Var$idx), !dbg !$loc
  ret void, !dbg !$loc
}
EOF
cat_mir_funcs <<EOF
---
name:            Fun$idx
tracksRegLiveness: true
registers:
  - { id: 0, class: gr64, preferred-register: '' }
stack:
  - { id: 0, name: Var$idx, type: default, offset: 0, size: $size, alignment: $align }
body:             |
  bb.0.entry:
    ; !DIExpr($expr)
    DBG_DEF !$lifetime, %stack.0
    RET64 debug-location !$loc
...
EOF
inc_idx
}

gencase_heterogeneous_mir_no_alloca_dbg_value() {
local referrer="$1"; shift
local expr="$1"; shift
declare_one_var_metadata
declare_mdid lifetime
cat_ir_metadata <<EOF
!$lifetime = distinct !DILifetime(object: !$var, location: !DIExpr($expr))
EOF
cat_ir_funcs <<EOF
define dso_local void @Fun$idx() #0 !dbg !$sub {
entry:
  ret void, !dbg !$loc
}
EOF
cat_mir_funcs <<EOF
---
name:            Fun$idx
tracksRegLiveness: true
registers:
  - { id: 0, class: gr64, preferred-register: '' }
stack: []
body:             |
  bb.0.entry:
    ; !DIExpr($expr)
    DBG_DEF !$lifetime, $referrer, debug-location !$loc
    RET64 debug-location !$loc
...
EOF
inc_idx
}

# Spit out common part of final test file
cat <<EOF
# NOTE: This file was generated by llvm/utils/gen-heterogeneous-debug-test.sh
# NOTE: Do not edit this file manually. Instead run:
# NOTE: llvm/utils/gen-heterogeneous-debug-test.sh > llvm/test/CodeGen/X86/heterogeneous-debug.test

# RUN: split-file %s %t

EOF

# BEGIN ir tests

reset_per_test_state
di_version=3

for i in "${!scalar_tys[@]}"; do
add_checks_ir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]})"
gencase_ir_one_alloca "${scalar_tys[$i]}" ''

add_checks_ir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_deref)"
gencase_ir_one_alloca "${scalar_tys[$i]}" 'DW_OP_deref'
done

cat <<EOF
;--- ir
; RUN: llc -O0 --filetype=obj < %t/ir | llvm-dwarfdump --diff --debug-info -name Var* -regex - | FileCheck %t/ir
EOF
print_ir_module

# END ir tests


# BEGIN mir tests

reset_per_test_state
di_version=3

for i in "${!scalar_tys[@]}"; do

add_checks_mir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]})"
gencase_mir_one_alloca "$i" ''

add_checks_mir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_deref)"
gencase_mir_one_alloca "$i" 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_stack_value)"
gencase_mir_one_alloca "$i" 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_fbreg -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_one_alloca "$i" 'DW_OP_deref, DW_OP_stack_value'

done


add_checks_mir "DW_AT_location (DW_OP_reg0 RAX)"
gencase_mir_no_alloca_dbg_values \$rax \$noreg ''

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0)"
gencase_mir_no_alloca_dbg_values \$rax \$noreg 'DW_OP_deref'

# This is a vexing cases, as there is essentially an implied DW_OP_deref (which
# is folded into a DW_OP_breg to make it valid DWARF)
add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$rax \$noreg 'DW_OP_stack_value'

# This illucidates the previous case somewhat: the presense of DW_OP_stack_value
# acts "at a distance" on the interpretation of the first 2 DBG_VALUE arguments.
add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$rax \$noreg 'DW_OP_deref, DW_OP_stack_value'


add_checks_mir "DW_AT_location (DW_OP_reg0 RAX)"
gencase_mir_no_alloca_dbg_values \$ax \$noreg ''

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and)"
gencase_mir_no_alloca_dbg_values \$ax \$noreg 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$ax \$noreg 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$ax \$noreg 'DW_OP_deref, DW_OP_stack_value'


add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0)"
gencase_mir_no_alloca_dbg_values \$ax 0 ''

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and, DW_OP_deref)"
gencase_mir_no_alloca_dbg_values \$ax 0 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$ax 0 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_constu 0xffff, DW_OP_and, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$ax 0 'DW_OP_deref, DW_OP_stack_value'


add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0)"
gencase_mir_no_alloca_dbg_values \$rax 0 ''

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_deref)"
gencase_mir_no_alloca_dbg_values \$rax 0 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$rax 0 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_breg0 RAX+0, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values \$rax 0 'DW_OP_deref, DW_OP_stack_value'


add_checks_mir "DW_AT_const_value (42)"
gencase_mir_no_alloca_dbg_values 42 \$noreg ''

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a)"
gencase_mir_no_alloca_dbg_values 42 \$noreg 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values 42 \$noreg 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values 42 \$noreg 'DW_OP_deref, DW_OP_stack_value'


# indirection ignored for const DBG_VALUE argument?
add_checks_mir "DW_AT_const_value (42)"
gencase_mir_no_alloca_dbg_values 42 0 ''

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a)"
gencase_mir_no_alloca_dbg_values 42 0 'DW_OP_deref'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values 42 0 'DW_OP_stack_value'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_deref, DW_OP_stack_value)"
gencase_mir_no_alloca_dbg_values 42 0 'DW_OP_deref, DW_OP_stack_value'


add_checks_mir \
  "DW_AT_location (indexed (0x[[#%x,]]) loclist = 0x[[#%x,]]:" \
  "[0x[[#%x,]], 0x[[#%x,]]): DW_OP_reg0 RAX, DW_OP_piece 0x4, DW_OP_reg3 RBX, DW_OP_piece 0x4)"
gencase_mir_no_alloca_dbg_values \
  \$rax \$noreg 'DW_OP_LLVM_fragment, 0, 32' \
  \$rbx \$noreg 'DW_OP_LLVM_fragment, 32, 32'

add_checks_mir \
  "DW_AT_location (indexed (0x[[#%x,]]) loclist = 0x[[#%x,]]:" \
  "[0x[[#%x,]], 0x[[#%x,]]): DW_OP_breg0 RAX+0, DW_OP_piece 0x4, DW_OP_reg3 RBX, DW_OP_piece 0x4)"
gencase_mir_no_alloca_dbg_values \
  \$rax 0 'DW_OP_LLVM_fragment, 0, 32' \
  \$rbx \$noreg 'DW_OP_LLVM_fragment, 32, 32'

# The YAML parser requires this all be indented
ir_module="$(print_ir_module | sed 's/^/  /')"
cat <<EOF
#--- mir
# RUN: llc -x mir -O0 -start-after=x86-isel -filetype=obj < %t/mir | llvm-dwarfdump --diff --debug-info -name Var* -regex - | FileCheck %t/mir
--- |
$ir_module
...
$mir_funcs
EOF

# END mir tests


# BEGIN heterogeneous_mir tests

reset_per_test_state
di_version=4

for i in "${!scalar_tys[@]}"; do

add_checks_mir "DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_consts -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_LLVM_user DW_OP_LLVM_offset"
gencase_heterogeneous_mir_one_alloca "$i" "DIOpReferrer(${scalar_tys[$i]})"

add_checks_mir "DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_consts -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address"
gencase_heterogeneous_mir_one_alloca "$i" "DIOpReferrer(ptr), DIOpDeref(${scalar_tys[$i]})"

deref_and_mask_ops="DW_OP_deref_size 0x$(printf '%x' "${scalar_ty_byte_sizes[$i]}")"
if [[ "${scalar_ty_bit_masks[$i]}" -ne 0 ]]; then
  deref_and_mask_ops+=", DW_OP_constu 0x$(printf '%x' "${scalar_ty_bit_masks[$i]}"), DW_OP_and"
fi

add_checks_mir "DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_consts -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_LLVM_user DW_OP_LLVM_offset, $deref_and_mask_ops, DW_OP_stack_value"
gencase_heterogeneous_mir_one_alloca "$i" "DIOpReferrer(${scalar_tys[$i]}), DIOpRead()"

add_checks_mir "DW_OP_reg6 RBP, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_consts -${scalar_ty_pow2_byte_sizes[$i]}, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, $deref_and_mask_ops, DW_OP_stack_value"
gencase_heterogeneous_mir_one_alloca "$i" "DIOpReferrer(ptr), DIOpDeref(${scalar_tys[$i]}), DIOpRead()"

done


add_checks_mir "DW_AT_location (DW_OP_reg0 RAX)"
gencase_heterogeneous_mir_no_alloca_dbg_value \$rax 'DIOpReferrer(i64)'

add_checks_mir "DW_AT_location (DW_OP_reg0 RAX, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)"
gencase_heterogeneous_mir_no_alloca_dbg_value \$rax 'DIOpReferrer(ptr), DIOpDeref(i64)'

add_checks_mir "DW_AT_location (DW_OP_reg0 RAX, DW_OP_deref_size 0x8, DW_OP_stack_value"
gencase_heterogeneous_mir_no_alloca_dbg_value \$rax 'DIOpReferrer(i64), DIOpRead()'

add_checks_mir "DW_AT_location (DW_OP_reg0 RAX, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_deref_size 0x8, DW_OP_stack_value)"
gencase_heterogeneous_mir_no_alloca_dbg_value \$rax 'DIOpReferrer(ptr), DIOpDeref(i64), DIOpRead()'


# FIXME: Need to locate subregs by offset?
#add_checks_mir "DW_AT_location (DW_OP_reg0 RAX)"
#gencase_heterogeneous_mir_no_alloca_dbg_value \$ax 'DIOpReferrer(i16)'


add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value)"
gencase_heterogeneous_mir_no_alloca_dbg_value 42 'DIOpReferrer(i64)'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)"
gencase_heterogeneous_mir_no_alloca_dbg_value 42 'DIOpReferrer(ptr), DIOpDeref(i64)'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_stack_value)"
gencase_heterogeneous_mir_no_alloca_dbg_value 42 'DIOpReferrer(i64), DIOpRead()'

add_checks_mir "DW_AT_location (DW_OP_constu 0x2a, DW_OP_stack_value, DW_OP_deref_size 0x8, DW_OP_constu 0x0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address, DW_OP_deref_size 0x8, DW_OP_stack_value)"
gencase_heterogeneous_mir_no_alloca_dbg_value 42 'DIOpReferrer(ptr), DIOpDeref(i64), DIOpRead()'


# The YAML parser requires this all be indented
ir_module="$(print_ir_module | sed 's/^/  /')"
cat <<EOF
#--- heterogeneous_mir
# RUN: llc -x mir -O0 -start-after=x86-isel -filetype=obj < %t/heterogeneous_mir | llvm-dwarfdump --diff --debug-info -name Var* -regex - | FileCheck %t/heterogeneous_mir
--- |
$ir_module
...
$mir_funcs
EOF

# END heterogeneous_mir tests
