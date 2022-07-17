; RUN: opt %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline-act.prof

; Sample profile should have non-empty ACT passed to inliner

; int t;
; bool foo(int value) {
;   switch(value) {
;     case 0:
;     case 1:
;     case 3:
;       return true;
;     default:
;       return false;
;   }
; }
; void bar(int i) {
;   if (foo(i))
;     t *= 2;
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@t = global i32 0, align 4

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z3fooi(i32) #0 {
  %switch.tableidx = sub i32 %0, 0
  %2 = icmp ult i32 %switch.tableidx, 4
  br i1 %2, label %switch.lookup, label %3

switch.lookup:                                    ; preds = %1
  %switch.cast = trunc i32 %switch.tableidx to i4
  %switch.shiftamt = mul i4 %switch.cast, 1
  %switch.downshift = lshr i4 -5, %switch.shiftamt
  %switch.masked = trunc i4 %switch.downshift to i1
  ret i1 %switch.masked

; <label>:3:                                      ; preds = %1
  ret i1 false
}

; Function Attrs: nounwind uwtable
define void @_Z3bari(i32) #0 !dbg !9 {
  %2 = call zeroext i1 @_Z3fooi(i32 %0), !dbg !10
  br i1 %2, label %3, label %6, !dbg !10

; <label>:3:                                      ; preds = %1
  %4 = load i32, i32* @t, align 4
  %5 = shl nsw i32 %4, 1
  store i32 %5, i32* @t, align 4
  br label %6

; <label>:6:                                      ; preds = %3, %1
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272227) (llvm/trunk 272226)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "test.cc", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang version 3.9.0 (trunk 272227) (llvm/trunk 272226)"}
!6 = !DISubroutineType(types: !2)
!9 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 14, type: !6, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!10 = !DILocation(line: 15, column: 7, scope: !9)
!11 = !DILocation(line: 16, column: 7, scope: !9)
