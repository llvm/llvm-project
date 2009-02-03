; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; This file contains the output from the following compiled C code:
; typedef struct list {
;   struct list *Next;
;   int Data;
; } list;
;
; // Iterative insert fn
; void InsertIntoListTail(list **L, int Data) {
;   while (*L)
;     L = &(*L)->Next;
;   *L = (list*)malloc(sizeof(list));
;   (*L)->Data = Data;
;   (*L)->Next = 0;
; }
;
; // Recursive list search fn
; list *FindData(list *L, int Data) {
;   if (L == 0) return 0;
;   if (L->Data == Data) return L;
;   return FindData(L->Next, Data);
; }
;
; void DoListStuff() {
;   list *MyList = 0;
;   InsertIntoListTail(&MyList, 100);
;   InsertIntoListTail(&MyList, 12);
;   InsertIntoListTail(&MyList, 42);
;   InsertIntoListTail(&MyList, 1123);
;   InsertIntoListTail(&MyList, 1213);
;
;   if (FindData(MyList, 75)) foundIt();
;   if (FindData(MyList, 42)) foundIt();
;   if (FindData(MyList, 700)) foundIt();
; }

%list = type { %list*, i32 }

declare i8* @malloc(i32)

define void @InsertIntoListTail(%list** %L, i32 %Data) {
bb1:
        %reg116 = load %list** %L               ; <%list*> [#uses=1]
        %cast1004 = inttoptr i64 0 to %list*            ; <%list*> [#uses=1]
        %cond1000 = icmp eq %list* %reg116, %cast1004           ; <i1> [#uses=1]
        br i1 %cond1000, label %bb3, label %bb2

bb2:            ; preds = %bb2, %bb1
        %reg117 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]           ; <%list**> [#uses=1]
        %cast1010 = bitcast %list** %reg117 to %list***         ; <%list***> [#uses=1]
        %reg118 = load %list*** %cast1010               ; <%list**> [#uses=3]
        %reg109 = load %list** %reg118          ; <%list*> [#uses=1]
        %cast1005 = inttoptr i64 0 to %list*            ; <%list*> [#uses=1]
        %cond1001 = icmp ne %list* %reg109, %cast1005           ; <i1> [#uses=1]
        br i1 %cond1001, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb1
        %reg119 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]           ; <%list**> [#uses=1]
        %cast1006 = bitcast %list** %reg119 to i8**             ; <i8**> [#uses=1]
        %reg111 = call i8* @malloc( i32 16 )            ; <i8*> [#uses=3]
        store i8* %reg111, i8** %cast1006
        %reg111.upgrd.1 = ptrtoint i8* %reg111 to i64           ; <i64> [#uses=1]
        %reg1002 = add i64 %reg111.upgrd.1, 8           ; <i64> [#uses=1]
        %reg1002.upgrd.2 = inttoptr i64 %reg1002 to i8*         ; <i8*> [#uses=1]
        %cast1008 = bitcast i8* %reg1002.upgrd.2 to i32*                ; <i32*> [#uses=1]
        store i32 %Data, i32* %cast1008
        %cast1003 = inttoptr i64 0 to i64*              ; <i64*> [#uses=1]
        %cast1009 = bitcast i8* %reg111 to i64**                ; <i64**> [#uses=1]
        store i64* %cast1003, i64** %cast1009
        ret void
}

define %list* @FindData(%list* %L, i32 %Data) {
bb1:
        br label %bb2

bb2:            ; preds = %bb6, %bb1
        %reg115 = phi %list* [ %reg116, %bb6 ], [ %L, %bb1 ]            ; <%list*> [#uses=4]
        %cast1014 = inttoptr i64 0 to %list*            ; <%list*> [#uses=1]
        %cond1011 = icmp ne %list* %reg115, %cast1014           ; <i1> [#uses=1]
        br i1 %cond1011, label %bb4, label %bb3

bb3:            ; preds = %bb2
        ret %list* null

bb4:            ; preds = %bb2
        %idx = getelementptr %list* %reg115, i64 0, i32 1               ; <i32*> [#uses=1]
        %reg111 = load i32* %idx                ; <i32> [#uses=1]
        %cond1013 = icmp ne i32 %reg111, %Data          ; <i1> [#uses=1]
        br i1 %cond1013, label %bb6, label %bb5

bb5:            ; preds = %bb4
        ret %list* %reg115

bb6:            ; preds = %bb4
        %idx2 = getelementptr %list* %reg115, i64 0, i32 0              ; <%list**> [#uses=1]
        %reg116 = load %list** %idx2            ; <%list*> [#uses=1]
        br label %bb2
}

