; The global symbol should be legalized
; RUN: llvm-as < %s | llc -march=alpha 

target datalayout = "e-p:64:64"
        %struct.LIST_HELP = type { %struct.LIST_HELP*, i8* }
        %struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [44 x i8] }
        %struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
@clause_SORT = external global [21 x %struct.LIST_HELP*]                ; <[21 x %struct.LIST_HELP*]*> [#uses=0]
@ia_in = external global %struct._IO_FILE*              ; <%struct._IO_FILE**> [#uses=1]
@multvec_j = external global [100 x i32]                ; <[100 x i32]*> [#uses=0]

define void @main(i32 %argc) {
clock_Init.exit:
        %tmp.5.i575 = load i32* null            ; <i32> [#uses=1]
        %tmp.309 = icmp eq i32 %tmp.5.i575, 0           ; <i1> [#uses=1]
        br i1 %tmp.309, label %UnifiedReturnBlock, label %then.17

then.17:                ; preds = %clock_Init.exit
        store %struct._IO_FILE* null, %struct._IO_FILE** @ia_in
        %savedstack = call i8* @llvm.stacksave( )               ; <i8*> [#uses=0]
        ret void

UnifiedReturnBlock:             ; preds = %clock_Init.exit
        ret void
}

declare i8* @llvm.stacksave()
