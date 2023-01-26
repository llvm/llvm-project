; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

@data = global [8193 x i32] zeroinitializer

define void @m(ptr %p) nounwind {
entry:
  ; CHECK-LABEL: m:

  call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(ptr elementtype(i32) @data)

  ; CHECK: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 0($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @m_offset_4(ptr %p) nounwind {
entry:
  ; CHECK-LABEL: m_offset_4:

  call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(ptr elementtype(i32) getelementptr inbounds ([8193 x i32], ptr @data, i32 0, i32 1))

  ; CHECK: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 4($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @m_offset_32764(ptr %p) nounwind {
entry:
  ; CHECK-LABEL: m_offset_32764:

  call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(ptr elementtype(i32) getelementptr inbounds ([8193 x i32], ptr @data, i32 0, i32 8191))

  ; CHECK-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK: #APP
  ; CHECK: lw $1, 32764($[[BASEPTR]])
  ; CHECK: #NO_APP

  ret void
}

define void @m_offset_32768(ptr %p) nounwind {
entry:
  ; CHECK-LABEL: m_offset_32768:

  call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(ptr elementtype(i32) getelementptr inbounds ([8193 x i32], ptr @data, i32 0, i32 8192))

  ; CHECK-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; CHECK-DAG: ori $[[T0:[0-9]+]], $zero, 32768
  ; CHECK: addu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], $[[T0]]
  ; CHECK: #APP
  ; CHECK: lw $1, 0($[[BASEPTR2]])
  ; CHECK: #NO_APP

  ret void
}
