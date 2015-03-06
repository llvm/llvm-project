; RUN: llc -march=mips -relocation-model=static -mattr=+noabicalls < %s | FileCheck %s

define i32* @get_gp() {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !0)
  %1 = inttoptr i32 %0 to i32*
  ret i32* %1
}

; CHECK-LABEL: get_gp:
; CHECK:           move $2, $gp

declare i32 @llvm.read_register.i32(metadata)

!llvm.named.register.$28 = !{!0}

!0 = metadata !{metadata !"$28"}
