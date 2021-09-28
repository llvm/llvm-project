; RUN: llc -mtriple=mips64 -relocation-model=static -mattr=+noabicalls -target-abi n32 < %s | FileCheck %s

define i32* @get_gp() {
entry:
  %0 = call i64 @llvm.read_register.i64(metadata !0)
  %1 = trunc i64 %0 to i32
  %2 = inttoptr i32 %1 to i32*
  ret i32* %2
}

; CHECK-LABEL: get_gp:
; CHECK:           sll $2, $gp, 0

declare i64 @llvm.read_register.i64(metadata)

!llvm.named.register.$28 = !{!0}

!0 = !{!"$28"}
