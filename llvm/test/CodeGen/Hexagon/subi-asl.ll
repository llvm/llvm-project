; RUN: llc -march=hexagon < %s | FileCheck %s

; Check if S4_subi_asl_ri is being generated correctly.

; CHECK-LABEL: yes_sub_asl
; FIXME: We no longer get subi_asl here. 
; XCHECK: [[REG1:(r[0-9]+)]] = sub(#0,asl([[REG1]],#1))
; CHECK: [[REG1:(r[0-9]+)]] = asl([[REG1]],#1)
; CHECK:  = sub(#0,[[REG1]])

; CHECK-LABEL: no_sub_asl
; CHECK: [[REG2:(r[0-9]+)]] = asl(r{{[0-9]+}},#1)
; CHECK: r{{[0-9]+}} = sub([[REG2]],r{{[0-9]+}})

%struct.rtx_def = type { i16, i8 }

@this_insn_number = external global i32, align 4

; Function Attrs: nounwind
define void @yes_sub_asl(%struct.rtx_def* %reg, %struct.rtx_def* nocapture readonly %setter) #0 {
entry:
  %code = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %reg, i32 0, i32 0
  %0 = load i16, i16* %code, align 4
  switch i16 %0, label %return [
    i16 2, label %if.end
    i16 5, label %if.end
  ]

if.end:
  %code6 = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %setter, i32 0, i32 0
  %1 = load i16, i16* %code6, align 4
  %cmp8 = icmp eq i16 %1, 56
  %conv9 = zext i1 %cmp8 to i32
  %2 = load i32, i32* @this_insn_number, align 4
  %3 = mul i32 %2, -2
  %sub = add nsw i32 %conv9, %3
  tail call void @reg_is_born(%struct.rtx_def* nonnull %reg, i32 %sub) #2
  br label %return

return:
  ret void
}

declare void @reg_is_born(%struct.rtx_def*, i32) #1

; Function Attrs: nounwind
define void @no_sub_asl(%struct.rtx_def* %reg, %struct.rtx_def* nocapture readonly %setter) #0 {
entry:
  %code = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %reg, i32 0, i32 0
  %0 = load i16, i16* %code, align 4
  switch i16 %0, label %return [
    i16 2, label %if.end
    i16 5, label %if.end
  ]

if.end:
  %1 = load i32, i32* @this_insn_number, align 4
  %mul = mul nsw i32 %1, 2
  %code6 = getelementptr inbounds %struct.rtx_def, %struct.rtx_def* %setter, i32 0, i32 0
  %2 = load i16, i16* %code6, align 4
  %cmp8 = icmp eq i16 %2, 56
  %conv9 = zext i1 %cmp8 to i32
  %sub = sub nsw i32 %mul, %conv9
  tail call void @reg_is_born(%struct.rtx_def* nonnull %reg, i32 %sub) #2
  br label %return

return:
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { "target-cpu"="hexagonv5" }
attributes #2 = { nounwind }
