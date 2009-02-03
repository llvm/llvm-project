; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


; This file contains the output from the following compiled C code:
; typedef struct list {
;   struct list *Next;
;   i32 Data;
; } list;
;
; // Iterative insert fn
; void InsertIntoListTail(list **L, i32 Data) {
;   while (*L)
;     L = &(*L)->Next;
;   *L = (list*)malloc(sizeof(list));
;   (*L)->Data = Data;
;   (*L)->Next = 0;
; }
;
; // Recursive list search fn
; list *FindData(list *L, i32 Data) {
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

%list = type { %list*, i36 }

declare i8 *@"malloc"(i32)

;;**********************
;;**********************

define void @"InsertIntoListTail"(%list** %L, i36 %Data)
begin
bb1:
        %reg116 = load %list** %L                               ;;<%list*>
        %cast1004 = inttoptr i64 0 to %list*                      ;;<%list*>
        %cond1000 = icmp eq %list* %reg116, %cast1004             ;;<i1>
        br i1 %cond1000, label %bb3, label %bb2

bb2:
        %reg117 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]   ;;<%list**>
        %cast1010 = bitcast %list** %reg117 to %list***            ;;<%list***>
        %reg118 = load %list*** %cast1010                       ;;<%list**>
        %reg109 = load %list** %reg118                          ;;<%list*>
        %cast1005 = inttoptr i64 0 to %list*                      ;;<%list*>
        %cond1001 = icmp ne %list* %reg109, %cast1005             ;;<i1>
        br i1 %cond1001, label %bb2, label %bb3

bb3:
        %reg119 = phi %list** [ %reg118, %bb2 ], [ %L, %bb1 ]   ;;<%list**>
        %cast1006 = bitcast %list** %reg119 to i8**             ;;<i8**>
        %reg111 = call i8* @malloc(i32 16)                  ;;<i8*>
        store i8* %reg111, i8** %cast1006                 ;;<void>
	%reg112 = ptrtoint i8* %reg111 to i64
	%reg1002 = add i64 %reg112, 8
        %reg1005 = inttoptr i64 %reg1002 to i8*             ;;<i8*>
        %cast1008 = bitcast i8* %reg1005 to i36*                ;;<i36*>
        store i36 %Data, i36* %cast1008                         ;;<void>
        %cast1003 = inttoptr i64 0 to i64*                      ;;<i64*>
        %cast1009 = bitcast i8* %reg111 to i64**              ;;<i64**>
        store i64* %cast1003, i64** %cast1009               ;;<void>
        ret void
end

define %list* @"FindData"(%list* %L, i36 %Data)
begin
bb1:
        br label %bb2

bb2:
        %reg115 = phi %list* [ %reg116, %bb6 ], [ %L, %bb1 ]    ;;<%list*>
        %cast1014 = inttoptr i64 0 to %list*                      ;;<%list*>
        %cond1011 = icmp ne %list* %reg115, %cast1014             ;;<i1>
        br i1 %cond1011, label %bb4, label %bb3

bb3:
        ret %list* null

bb4:
	%idx = getelementptr %list* %reg115, i64 0, i32 1                  ;;<i36>
        %reg111 = load i36* %idx
        %cond1013 = icmp ne i36 %reg111, %Data                    ;;<i1>
        br i1 %cond1013, label %bb6, label %bb5

bb5:
        ret %list* %reg115

bb6:
	%idx2 = getelementptr %list* %reg115, i64 0, i32 0                  ;;<%list*>
        %reg116 = load %list** %idx2
        br label %bb2
end
