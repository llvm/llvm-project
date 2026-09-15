; REQUIRES: x86-registered-target || aarch64-registered-target
; RUN: %if x86-registered-target %{ llc -mtriple=x86_64-pc-windows-msvc -x86-asm-syntax=intel -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,X86 %}
; RUN: %if x86-registered-target %{ llc -O0 -mtriple=x86_64-pc-windows-msvc -x86-asm-syntax=intel -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,X86 %}
; RUN: %if aarch64-registered-target %{ llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,AARCH64 %}
; RUN: %if aarch64-registered-target %{ llc -O0 -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,AARCH64 %}

declare void @may_fault(ptr)
declare i32 @filter(ptr, ptr)
declare void @llvm.seh.try.begin()
declare void @llvm.seh.try.end()
declare i32 @__C_specific_handler(...)

define void @leading_store(ptr %address) personality ptr @__C_specific_handler {
entry:
  store volatile i32 11, ptr %address
  invoke void @llvm.seh.try.begin()
          to label %guarded unwind label %dispatch

guarded:
  store volatile i32 22, ptr %address
  invoke void @may_fault(ptr %address)
          to label %guarded.end unwind label %dispatch

guarded.end:
  invoke void @llvm.seh.try.end()
          to label %exit unwind label %dispatch

dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller

handler:
  %pad = catchpad within %switch [ptr null]
  store volatile i32 44, ptr %address
  catchret from %pad to label %exit

exit:
  store volatile i32 33, ptr %address
  ret void
}

; CHECK-LABEL: leading_store:
; X86: mov dword ptr [{{[^]]+}}], 11
; AARCH64: mov [[BEFORE:w[0-9]+]], #11
; AARCH64-NEXT: str [[BEFORE]], [{{[^]]+}}]
; CHECK: %guarded{{$}}
; CHECK-NEXT: [[BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 22
; AARCH64: mov [[GUARDED:w[0-9]+]], #22
; AARCH64-NEXT: str [[GUARDED]], [{{[^]]+}}]
; CHECK: {{(call|bl)}} may_fault
; CHECK: [[END:\.Ltmp[0-9]+]]:
; CHECK: [[HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%handler{{$}}
; X86: mov dword ptr [{{[^]]+}}], 44
; AARCH64: mov [[CAUGHT:w[0-9]+]], #44
; AARCH64-NEXT: str [[CAUGHT]], [{{[^]]+}}]
; X86: mov dword ptr [{{[^]]+}}], 33
; AARCH64: mov [[AFTER:w[0-9]+]], #33
; AARCH64-NEXT: str [[AFTER]], [{{[^]]+}}]
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[HANDLER]]@IMGREL
; CHECK-NEXT: .Llsda_end{{[0-9]+}}:

define void @branching_handler(ptr %address, i1 %condition) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin()
          to label %guarded unwind label %dispatch

guarded:
  invoke void @may_fault(ptr %address)
          to label %guarded.end unwind label %dispatch

guarded.end:
  invoke void @llvm.seh.try.end()
          to label %exit unwind label %dispatch

dispatch:
  %switch = catchswitch within none [label %handler] unwind to caller

handler:
  %pad = catchpad within %switch [ptr @filter]
  store volatile i32 44, ptr %address
  br i1 %condition, label %handler.left, label %handler.right

handler.left:
  store volatile i32 55, ptr %address
  catchret from %pad to label %exit

handler.right:
  store volatile i32 66, ptr %address
  catchret from %pad to label %exit

exit:
  store volatile i32 33, ptr %address
  ret void
}

; CHECK-LABEL: branching_handler:
; CHECK: %guarded{{$}}
; CHECK-NEXT: [[BRANCH_BEGIN:\.Ltmp[0-9]+]]:
; CHECK: {{(call|bl)}} may_fault
; CHECK: [[BRANCH_END:\.Ltmp[0-9]+]]:
; CHECK: [[BRANCH_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%handler{{$}}
; X86: mov dword ptr [{{[^]]+}}], 44
; AARCH64: mov [[BRANCH_CAUGHT:w[0-9]+]], #44
; AARCH64-NEXT: str [[BRANCH_CAUGHT]], [{{[^]]+}}]
; X86: mov dword ptr [{{[^]]+}}], 55
; AARCH64: mov [[LEFT:w[0-9]+]], #55
; AARCH64-NEXT: str [[LEFT]], [{{[^]]+}}]
; X86: mov dword ptr [{{[^]]+}}], 66
; AARCH64: mov [[RIGHT:w[0-9]+]], #66
; AARCH64-NEXT: str [[RIGHT]], [{{[^]]+}}]
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[BRANCH_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[BRANCH_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} filter@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[BRANCH_HANDLER]]@IMGREL
; CHECK-NEXT: .Llsda_end{{[0-9]+}}:

define void @nested_handler(ptr %address, i1 %condition) personality ptr @__C_specific_handler {
entry:
  invoke void @llvm.seh.try.begin()
          to label %outer unwind label %outer.dispatch

outer:
  invoke void @llvm.seh.try.begin()
          to label %middle unwind label %middle.dispatch

middle:
  invoke void @llvm.seh.try.begin()
          to label %inner unwind label %inner.dispatch

inner:
  invoke void @may_fault(ptr %address)
          to label %inner.end unwind label %inner.dispatch

inner.end:
  invoke void @llvm.seh.try.end()
          to label %middle.end unwind label %inner.dispatch

middle.end:
  store volatile i32 77, ptr %address
  invoke void @llvm.seh.try.end()
          to label %outer.end unwind label %middle.dispatch

outer.end:
  store volatile i32 88, ptr %address
  invoke void @llvm.seh.try.end()
          to label %exit unwind label %outer.dispatch

inner.dispatch:
  %inner.switch = catchswitch within none [label %inner.handler] unwind label %middle.dispatch

inner.handler:
  %inner.pad = catchpad within %inner.switch [ptr null]
  store volatile i32 44, ptr %address
  br i1 %condition, label %inner.left, label %inner.right

inner.left:
  store volatile i32 55, ptr %address
  catchret from %inner.pad to label %middle.end

inner.right:
  store volatile i32 66, ptr %address
  catchret from %inner.pad to label %middle.end

middle.dispatch:
  %middle.switch = catchswitch within none [label %middle.handler] unwind label %outer.dispatch

middle.handler:
  %middle.pad = catchpad within %middle.switch [ptr null]
  store volatile i32 99, ptr %address
  catchret from %middle.pad to label %outer.end

outer.dispatch:
  %outer.switch = catchswitch within none [label %outer.handler] unwind to caller

outer.handler:
  %outer.pad = catchpad within %outer.switch [ptr null]
  store volatile i32 111, ptr %address
  catchret from %outer.pad to label %exit

exit:
  store volatile i32 33, ptr %address
  ret void
}

; CHECK-LABEL: nested_handler:
; CHECK: %inner{{$}}
; CHECK-NEXT: [[INNER_BEGIN:\.Ltmp[0-9]+]]:
; CHECK: {{(call|bl)}} may_fault
; CHECK: [[INNER_END:\.Ltmp[0-9]+]]:
; CHECK: %middle.end{{$}}
; CHECK-NEXT: [[MIDDLE_BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 77
; AARCH64: mov [[MIDDLE_VALUE:w[0-9]+]], #77
; AARCH64-NEXT: str [[MIDDLE_VALUE]], [{{[^]]+}}]
; CHECK-NEXT: [[MIDDLE_END:\.Ltmp[0-9]+]]:
; CHECK: %outer.end{{$}}
; CHECK-NEXT: [[OUTER_BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 88
; AARCH64: mov [[OUTER_VALUE:w[0-9]+]], #88
; AARCH64-NEXT: str [[OUTER_VALUE]], [{{[^]]+}}]
; CHECK-NEXT: [[OUTER_END:\.Ltmp[0-9]+]]:
; CHECK: [[INNER_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%inner.handler{{$}}
; CHECK: [[INNER_HANDLER_BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 44
; AARCH64: mov [[INNER_CAUGHT:w[0-9]+]], #44
; AARCH64-NEXT: str [[INNER_CAUGHT]], [{{[^]]+}}]
; X86: mov dword ptr [{{[^]]+}}], 55
; AARCH64: mov [[INNER_LEFT:w[0-9]+]], #55
; AARCH64-NEXT: str [[INNER_LEFT]], [{{[^]]+}}]
; X86: mov dword ptr [{{[^]]+}}], 66
; AARCH64: mov [[INNER_RIGHT:w[0-9]+]], #66
; AARCH64-NEXT: str [[INNER_RIGHT]], [{{[^]]+}}]
; CHECK-NEXT: [[INNER_HANDLER_END:\.Ltmp[0-9]+]]:
; CHECK: [[MIDDLE_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%middle.handler{{$}}
; CHECK: [[MIDDLE_HANDLER_BEGIN:\.Ltmp[0-9]+]]:
; X86: mov dword ptr [{{[^]]+}}], 99
; AARCH64: mov [[MIDDLE_CAUGHT:w[0-9]+]], #99
; AARCH64-NEXT: str [[MIDDLE_CAUGHT]], [{{[^]]+}}]
; CHECK-NEXT: [[MIDDLE_HANDLER_END:\.Ltmp[0-9]+]]:
; CHECK: [[OUTER_HANDLER:\.LBB[0-9]+_[0-9]+]]: {{.*}}%outer.handler{{$}}
; X86: mov dword ptr [{{[^]]+}}], 111
; AARCH64: mov [[OUTER_CAUGHT:w[0-9]+]], #111
; AARCH64-NEXT: str [[OUTER_CAUGHT]], [{{[^]]+}}]
; CHECK: .Llsda_begin{{[0-9]+}}:
; CHECK-NEXT: {{\.(long|word)}} [[INNER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[INNER_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_HANDLER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_HANDLER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_HANDLER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[INNER_HANDLER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_HANDLER]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_HANDLER_BEGIN]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} [[MIDDLE_HANDLER_END]]@IMGREL
; CHECK-NEXT: {{\.(long|word)}} 1
; CHECK-NEXT: {{\.(long|word)}} [[OUTER_HANDLER]]@IMGREL
; CHECK-NEXT: .Llsda_end{{[0-9]+}}:

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"eh-asynch", i32 1}