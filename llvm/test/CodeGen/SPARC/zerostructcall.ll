; RUN: llc < %s -march=sparc | FileCheck %s

; CHECK-LABEL: struct_ptr_test
; CHECK:       call struct_ptr_fn
; CHECK-NEXT:  st %i0, [%fp+-4]
; CHECK-NEXT:  ret

%struct.S = type {}

define void @struct_ptr_test(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = bitcast i32* %i.addr to %struct.S*
  call void @struct_ptr_fn(%struct.S* byval align 1 %0)
  ret void
}

declare void @struct_ptr_fn(%struct.S* byval align 1)

; CHECK-LABEL: struct_test
; CHECK:       call struct_fn
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret

%struct.U = type {}

@a = internal global [1 x %struct.U] zeroinitializer, align 1

define void @struct_test() {
entry:
  tail call void @struct_fn(%struct.U* byval align 1 getelementptr inbounds ([1 x %struct.U], [1 x %struct.U]* @a, i32 0, i32 0))
  ret void
}

; CHECK-LABEL: struct_arg_test
; CHECK:       call struct_arg_fn
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret

declare void @struct_fn(%struct.U* byval align 1)

@b = internal global [1 x %struct.U] zeroinitializer, align 1

define void @struct_arg_test() {
entry:
  tail call void @struct_arg_fn(%struct.U* byval align 1 getelementptr inbounds ([1 x %struct.U], [1 x %struct.U]* @b, i32 0, i32 0))
  ret void
}

declare void @struct_arg_fn(%struct.U* byval align 1)
