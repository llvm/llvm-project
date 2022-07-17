; RUN: opt -S -passes=canonicalize-aliases < %s | FileCheck %s
; RUN: opt -passes='thinlto-pre-link<O0>,require<module-summary>' -o - < %s | llvm-dis -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DAG: @analias = alias void (), void ()* @aliasee
; CHECK-DAG: @anotheralias = alias void (), void ()* @aliasee
; CHECK-DAG: define void @aliasee()

@analias = alias void (), void ()* @anotheralias
@anotheralias = alias void (), bitcast (void ()* @aliasee to void ()*)

; Function Attrs: nounwind uwtable
define void @aliasee() #0 {
entry:
    ret void
}

%struct.S1 = type { i32, i32, i32 }

; CHECK-DAG: @S = global %struct.S1 { i32 31, i32 32, i32 33 }
; CHECK-DAG: @Salias = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @S, i32 0, i32 1)
; CHECK-DAG: @Salias2 = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @S, i32 0, i32 1)
; CHECK-DAG: @Salias3 = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @S, i32 0, i32 1)

@S = global %struct.S1 { i32 31, i32 32, i32 33 }, align 4
@Salias = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @S, i32 0, i32 1)
@Salias2 = alias i32, i32* @Salias
@Salias3 = alias i32, i32* @Salias2

; CHECK-DAG: @Salias4 = alias %struct.S1, %struct.S1* @S
; CHECK-DAG: @Salias5 = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @S, i32 0, i32 1)

@Salias4 = alias %struct.S1, %struct.S1* @S
@Salias5 = alias i32, getelementptr inbounds (%struct.S1, %struct.S1* @Salias4, i32 0, i32 1)
