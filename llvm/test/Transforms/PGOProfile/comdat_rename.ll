; RUN: opt < %s -mtriple=x86_64-unknown-linux -pgo-instr-gen -do-comdat-renaming=true -S | FileCheck --check-prefixes COMMON,ELFONLY %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux -passes=pgo-instr-gen -do-comdat-renaming=true -S | FileCheck --check-prefixes COMMON,ELFONLY %s
; RUN: opt < %s -mtriple=x86_64-pc-win32-coff -pgo-instr-gen -do-comdat-renaming=true -S | FileCheck --check-prefixes COMMON,COFFONLY %s
; RUN: opt < %s -mtriple=x86_64-pc-win32-coff -passes=pgo-instr-gen -do-comdat-renaming=true -S | FileCheck --check-prefixes COMMON,COFFONLY %s

; Rename Comdat group and its function.
$f = comdat any
; COMMON: $f.[[SINGLEBB_HASH:[0-9]+]] = comdat any
define linkonce_odr void @f() comdat($f) {
  ret void
}

; Not rename Comdat with right linkage.
$nf = comdat any
; COMMON: $nf = comdat any
define void @nf() comdat($nf) {
  ret void
}

; Not rename Comdat with variable members.
$f_with_var = comdat any
; COMMON: $f_with_var = comdat any
@var = global i32 0, comdat($f_with_var)
define linkonce_odr void @f_with_var() comdat($f_with_var) {
  %tmp = load i32, i32* @var, align 4
  %inc = add nsw i32 %tmp, 1
  store i32 %inc, i32* @var, align 4
  ret void
}

; Not rename Comdat with multiple functions.
$tf = comdat any
; COMMON: $tf = comdat any
define linkonce void @tf() comdat($tf) {
  ret void
}
define linkonce void @tf2() comdat($tf) {
  ret void
}

; Rename AvailableExternallyLinkage functions
; ELFONLY-DAG: $aef.[[SINGLEBB_HASH]] = comdat any

; ELFONLY: @f = weak alias void (), void ()* @f.[[SINGLEBB_HASH]]
; ELFONLY: @aef = weak alias void (), void ()* @aef.[[SINGLEBB_HASH]]

define available_externally void @aef() {
; ELFONLY: define linkonce_odr void @aef.[[SINGLEBB_HASH]]() comdat {
; COFFONLY: define available_externally void @aef() {
  ret void
}


