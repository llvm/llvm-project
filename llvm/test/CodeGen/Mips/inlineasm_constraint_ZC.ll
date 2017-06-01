; RUN: llc -march=mipsel -mcpu=mips32r6 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,09BIT
; RUN: llc -march=mipsel -mattr=+micromips -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,12BIT
; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,16BIT

@data = global [8193 x i32] zeroinitializer

define void @ZC(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 0))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; ALL: #APP
  ; ALL: lw $1, 0($[[BASEPTR]])
  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_n4(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_n4:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 -1))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; ALL: #APP
  ; ALL: lw $1, -4($[[BASEPTR]])
  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_4(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_4:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 1))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; ALL: #APP
  ; ALL: lw $1, 4($[[BASEPTR]])
  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_252(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_252:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 63))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; ALL: #APP
  ; ALL: lw $1, 252($[[BASEPTR]])
  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_256(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_256:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 64))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(

  ; 09BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 256

  ; ALL: #APP

  ; 09BIT: lw $1, 0($[[BASEPTR2]])
  ; 12BIT: lw $1, 256($[[BASEPTR]])
  ; 16BIT: lw $1, 256($[[BASEPTR]])

  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_2044(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_2044:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 511))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(

  ; 09BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 2044

  ; ALL: #APP

  ; 09BIT: lw $1, 0($[[BASEPTR2]])
  ; 12BIT: lw $1, 2044($[[BASEPTR]])
  ; 16BIT: lw $1, 2044($[[BASEPTR]])

  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_2048(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_2048:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 512))

  ; ALL: lw $[[BASEPTR:[0-9]+]], %got(data)(

  ; 09BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 2048
  ; 12BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 2048

  ; ALL: #APP

  ; 09BIT: lw $1, 0($[[BASEPTR2]])
  ; 12BIT: lw $1, 0($[[BASEPTR2]])
  ; 16BIT: lw $1, 2048($[[BASEPTR]])

  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_32764(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_32764:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 8191))

  ; ALL-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(

  ; 09BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 32764
  ; 12BIT: addiu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], 32764

  ; ALL: #APP

  ; 09BIT: lw $1, 0($[[BASEPTR2]])
  ; 12BIT: lw $1, 0($[[BASEPTR2]])
  ; 16BIT: lw $1, 32764($[[BASEPTR]])

  ; ALL: #NO_APP

  ret void
}

define void @ZC_offset_32768(i32 *%p) nounwind {
entry:
  ; ALL-LABEL: ZC_offset_32768:

  call void asm sideeffect "lw $$1, $0", "*^ZC,~{$1}"(i32* getelementptr inbounds ([8193 x i32], [8193 x i32]* @data, i32 0, i32 8192))

  ; ALL-DAG: lw $[[BASEPTR:[0-9]+]], %got(data)(
  ; ALL-DAG: ori $[[T0:[0-9]+]], $zero, 32768

  ; 09BIT: addu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], $[[T0]]
  ; 12BIT: addu16 $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], $[[T0]]
  ; 16BIT: addu $[[BASEPTR2:[0-9]+]], $[[BASEPTR]], $[[T0]]

  ; ALL: #APP
  ; ALL: lw $1, 0($[[BASEPTR2]])
  ; ALL: #NO_APP

  ret void
}
