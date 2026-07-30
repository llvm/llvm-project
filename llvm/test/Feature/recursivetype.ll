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

%list = type { ptr, i32 }

declare ptr @malloc(i32)

define void @InsertIntoListTail(ptr %L, i32 %Data) {
bb1:
        %reg116 = load ptr, ptr %L               ; <ptr> [#uses=1]
        %cast1004 = inttoptr i64 0 to ptr            ; <ptr> [#uses=1]
        %cond1000 = icmp eq ptr %reg116, %cast1004           ; <i1> [#uses=1]
        br i1 %cond1000, label %bb3, label %bb2

bb2:            ; preds = %bb2, %bb1
        %reg117 = phi ptr [ %reg118, %bb2 ], [ %L, %bb1 ]           ; <ptr> [#uses=1]
        %reg118 = load ptr, ptr %reg117               ; <ptr> [#uses=3]
        %reg109 = load ptr, ptr %reg118          ; <ptr> [#uses=1]
        %cast1005 = inttoptr i64 0 to ptr            ; <ptr> [#uses=1]
        %cond1001 = icmp ne ptr %reg109, %cast1005           ; <i1> [#uses=1]
        br i1 %cond1001, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb1
        %reg119 = phi ptr [ %reg118, %bb2 ], [ %L, %bb1 ]           ; <ptr> [#uses=1]
        %reg111 = call ptr @malloc( i32 16 )            ; <ptr> [#uses=3]
        store ptr %reg111, ptr %reg119
        %reg111.upgrd.1 = ptrtoint ptr %reg111 to i64           ; <i64> [#uses=1]
        %reg1002 = add i64 %reg111.upgrd.1, 8           ; <i64> [#uses=1]
        %reg1002.upgrd.2 = inttoptr i64 %reg1002 to ptr         ; <ptr> [#uses=1]
        store i32 %Data, ptr %reg1002.upgrd.2
        %cast1003 = inttoptr i64 0 to ptr              ; <ptr> [#uses=1]
        store ptr %cast1003, ptr %reg111
        ret void
}

define ptr @FindData(ptr %L, i32 %Data) {
bb1:
        br label %bb2

bb2:            ; preds = %bb6, %bb1
        %reg115 = phi ptr [ %reg116, %bb6 ], [ %L, %bb1 ]            ; <ptr> [#uses=4]
        %cast1014 = inttoptr i64 0 to ptr            ; <ptr> [#uses=1]
        %cond1011 = icmp ne ptr %reg115, %cast1014           ; <i1> [#uses=1]
        br i1 %cond1011, label %bb4, label %bb3

bb3:            ; preds = %bb2
        ret ptr null

bb4:            ; preds = %bb2
        %idx = getelementptr %list, ptr %reg115, i64 0, i32 1               ; <ptr> [#uses=1]
        %reg111 = load i32, ptr %idx                ; <i32> [#uses=1]
        %cond1013 = icmp ne i32 %reg111, %Data          ; <i1> [#uses=1]
        br i1 %cond1013, label %bb6, label %bb5

bb5:            ; preds = %bb4
        ret ptr %reg115

bb6:            ; preds = %bb4
        %idx2 = getelementptr %list, ptr %reg115, i64 0, i32 0              ; <ptr> [#uses=1]
        %reg116 = load ptr, ptr %idx2            ; <ptr> [#uses=1]
        br label %bb2
}

