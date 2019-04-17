; RUN: opt < %s -simplifycfg -S | FileCheck %s

; CHECK-NOT: switch
        %llvm.dbg.anchor.type = type { i32, i32 }
        %llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8*, i1, i1, i8* }

@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"

@.str = internal constant [4 x i8] c"a.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"	; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [55 x i8] c"4.2.1 (Based on Apple Inc. build 5636) (LLVM build 00)\00", section "llvm.metadata"		; <[55 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([55 x i8], [55 x i8]* @.str2, i32 0, i32 0), i1 true, i1 false, i8* null }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]

declare void @llvm.dbg.stoppoint(i32, i32, { }*) nounwind

; Test folding all to same dest
define i32 @test3(i1 %C) {
        br i1 %C, label %Start, label %TheDest
Start:          ; preds = %0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        switch i32 3, label %TheDest [
                 i32 0, label %TheDest
                 i32 1, label %TheDest
                 i32 2, label %TheDest
                 i32 5, label %TheDest
        ]
TheDest:                ; preds = %Start, %Start, %Start, %Start, %Start, %0
        ret i32 1234
}

; Test folding switch -> branch
define i32 @test4(i32 %C) {
        switch i32 %C, label %L1 [
                 i32 0, label %L2
        ]
L1:             ; preds = %0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        ret i32 0
L2:             ; preds = %0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        ret i32 1
}

; Can fold into a cond branch!
define i32 @test5(i32 %C) {
        switch i32 %C, label %L1 [
                 i32 0, label %L2
                 i32 123, label %L1
        ]
L1:             ; preds = %0, %0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        ret i32 0
L2:             ; preds = %0
call void @llvm.dbg.stoppoint(i32 5, i32 0, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*))
        ret i32 1
}

