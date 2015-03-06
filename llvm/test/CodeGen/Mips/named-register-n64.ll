; RUN: llc -march=mips64 -relocation-model=static -mattr=+noabicalls < %s | FileCheck %s

define i32* @get_gp() {
entry:
  %0 = call i64 @llvm.read_register.i64(metadata !0)
  %1 = inttoptr i64 %0 to i32*
  ret i32* %1
}

; CHECK-LABEL: get_gp:
; CHECK:           move $2, $gp

declare i64 @llvm.read_register.i64(metadata)

!llvm.named.register.$28 = !{!0}

!0 = metadata !{metadata !"$28"}
