; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

@data = global [8193 x i32] zeroinitializer

define void @R(i32 *%p) nounwind {
entry:
  ; CHECK-LABEL: R:

  call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* elementtype(i32) getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 0))

  ; CHECK: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 0($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @R_offset_4(i32 *%p) nounwind {
entry:
  ; CHECK-LABEL: R_offset_4:

  call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* elementtype(i32) getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 1))

  ; CHECK: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 4($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @R_offset_254(i32 *%p) nounwind {
entry:
  ; CHECK-LABEL: R_offset_254:

  call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* elementtype(i32) getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 63))

  ; CHECK-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 252($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @R_offset_256(i32 *%p) nounwind {
entry:
  ; CHECK-LABEL: R_offset_256:

  call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* elementtype(i32) getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 64))

  ; CHECK-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 256
  ; CHECK: #APP
  ; CHECK: lw $1, 0($[[BASEPTR2]])
  ; CHECK: #NO_APP

  ret void
}
