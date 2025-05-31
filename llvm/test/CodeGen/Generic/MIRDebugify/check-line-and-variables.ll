; RUN: llc -debugify-check-and-strip-all-safe -o - %s 2>&1 | FileCheck %s
; RUN: llc --experimental-debuginfo-iterators=false -debugify-check-and-strip-all-safe -o - %s 2>&1 | FileCheck %s

; ModuleID = 'main.c'
source_filename = "main.c"

@ga = dso_local global i32 2, align 4

define dso_local i32 @foo(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %c, align 4
  %2 = load i32, ptr %c, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr @ga, align 4
  %3 = load i32, ptr %c, align 4
  ret i32 %3
}

; Different Back-Ends may have different number of passes, here we only
; check two of them to make sure -debugify-check-and-strip-all-safe works.
;CHECK: Machine IR debug info check: PASS
;CHECK: Machine IR debug info check: PASS
