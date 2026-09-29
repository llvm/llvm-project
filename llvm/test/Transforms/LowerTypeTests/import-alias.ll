; RUN: opt -S %s -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import-alias.yaml | FileCheck %s
;
; Check that the definitions for @f and @f_alias are removed from this module
; but @g_alias remains.
;
; CHECK: @f_dso_local_alias.cfi = dso_local alias void (), ptr @f_dso_local.cfi
; CHECK: @g_alias = alias void (), ptr @g
; CHECK: define hidden void @f.cfi()
; CHECK: define hidden void @f_dso_local.cfi()
; CHECK: declare void @f()
; CHECK: declare void @f_alias()
; CHECK: declare dso_local void @f_dso_local()
; CHECK: declare dso_local void @f_dso_local_alias()

target triple = "x86_64-unknown-linux"

@f_alias = alias void (), ptr @f
@f_dso_local_alias = dso_local alias void (), ptr @f_dso_local
@g_alias = alias void (), ptr @g

; Definition moved to the merged module
define void @f() {
  ret void
}

; Definition not moved to the merged module
define void @g() {
  ret void
}

define dso_local void @f_dso_local() {
  ret void
}

define void @uses_aliases() {
  call void @f_alias()
  call void @f_dso_local_alias()
  call void @g_alias()
  ret void
}
