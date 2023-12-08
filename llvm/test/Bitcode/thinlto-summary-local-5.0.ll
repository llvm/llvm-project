; Bitcode compatibility test for dso_local flag in thin-lto summaries.
; Checks that older bitcode summaries without the dso_local op are still
; properly parsed and don't set GlobalValues as dso_local.

; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: llvm-bcanalyzer -dump %s.bc | FileCheck %s --check-prefix=BCAN

define void @foo() {
;CHECK-DAG:define void @foo()
      ret void
}

@bar = global i32 0
;CHECK-DAG: @bar = global i32 0

@baz = alias i32, i32* @bar
;CHECK-DAG: @baz = alias i32, ptr @bar

;BCAN: <SOURCE_FILENAME
;BCAN-NEXT: <GLOBALVAR {{.*}} op7=0/>
;BCAN-NEXT: <FUNCTION {{.*}} op16=0/>
;BCAN-NEXT: <ALIAS {{.*}} op9=0/>
