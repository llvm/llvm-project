; ARMConstantIslands has to count the padding that aligns each funclet entry
; to the function's preferred alignment (WinException::beginFunclet), or a
; constant pool load across funclets can end up out of range.

; RUN: llc -mtriple=thumbv7-windows-msvc -stop-after=arm-cp-islands < %s | FileCheck %s

declare void @g(i32)
declare void @use(float)
declare void @dtor(ptr)
declare i32 @__C_specific_handler(...)

; A constant pool makes the function 4-byte aligned, and so its funclets.
; CHECK-LABEL: name: cp
; CHECK:       alignment: 4
; CHECK:       ehfunclet-entry{{.*}}, align 4):
define void @cp(float %v) personality ptr @__C_specific_handler {
entry:
  %obj = alloca i32
  invoke void @g(i32 1) to label %cont unwind label %cleanup
cont:
  %m = fmul float %v, 1.234500e+03
  call void @use(float %m)
  call void @dtor(ptr %obj)
  ret void
cleanup:
  %cp = cleanuppad within none []
  call void @dtor(ptr %obj) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; So does a jump table, which ARMConstantIslands finds only after it has
; measured the blocks.
; CHECK-LABEL: name: jt
; CHECK:       alignment: 4
; CHECK:       ehfunclet-entry{{.*}}, align 4):
define void @jt(i32 %v) personality ptr @__C_specific_handler {
entry:
  %obj = alloca i32
  invoke void @g(i32 0) to label %cont unwind label %cleanup
cont:
  switch i32 %v, label %done [
    i32 0, label %c0
    i32 1, label %c1
    i32 2, label %c2
    i32 3, label %c3
    i32 4, label %c4
    i32 5, label %c5
  ]
c0:
  call void @g(i32 10)
  br label %done
c1:
  call void @g(i32 11)
  br label %done
c2:
  call void @g(i32 12)
  br label %done
c3:
  call void @g(i32 13)
  br label %done
c4:
  call void @g(i32 14)
  br label %done
c5:
  call void @g(i32 15)
  br label %done
done:
  call void @dtor(ptr %obj)
  ret void
cleanup:
  %cp = cleanuppad within none []
  call void @dtor(ptr %obj) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; Without either, the funclet keeps the function's own alignment.
; CHECK-LABEL: name: nocp
; CHECK:       alignment: [[ALIGN:[0-9]+]]
; CHECK:       ehfunclet-entry{{.*}}, align [[ALIGN]]):
define void @nocp(i32 %v) personality ptr @__C_specific_handler {
entry:
  %obj = alloca i32
  invoke void @g(i32 %v) to label %cont unwind label %cleanup
cont:
  call void @dtor(ptr %obj)
  ret void
cleanup:
  %cp = cleanuppad within none []
  call void @dtor(ptr %obj) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}
