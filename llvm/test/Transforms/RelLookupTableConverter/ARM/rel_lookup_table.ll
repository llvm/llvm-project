; RUN: opt -passes=rel-lookup-table-converter                       -mtriple=arm-none-eabi -S < %s | FileCheck %s --check-prefix=ABS
; RUN: opt -passes=rel-lookup-table-converter -relocation-model=pic -mtriple=arm-none-eabi -S < %s | FileCheck %s --check-prefix=REL

@.str.0 = private unnamed_addr constant [6 x i8] c"Lorem\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"ipsum\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"dolor\00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"sit\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"amet\00", align 1
@.str.5 = private unnamed_addr constant [12 x i8] c"consectetur\00", align 1
@.str.6 = private unnamed_addr constant [12 x i8] c"adipisicing\00", align 1
@.str.default = private unnamed_addr constant [5 x i8] c"elit\00", align 1
@.table = private unnamed_addr constant [7 x ptr] [
  ptr @.str.0,
  ptr @.str.1,
  ptr @.str.2,
  ptr @.str.3,
  ptr @.str.4,
  ptr @.str.5,
  ptr @.str.6
], align 4
; ABS: @.table = private unnamed_addr constant [7 x ptr] [ptr @.str.0, ptr @.str.1, ptr @.str.2, ptr @.str.3, ptr @.str.4, ptr @.str.5, ptr @.str.6], align 4
; REL: @reltable.lookup = private unnamed_addr constant [7 x i32] [i32 sub (i32 ptrtoint (ptr @.str.0 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.1 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.2 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.3 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.4 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.5 to i32), i32 ptrtoint (ptr @reltable.lookup to i32)), i32 sub (i32 ptrtoint (ptr @.str.6 to i32), i32 ptrtoint (ptr @reltable.lookup to i32))], align 4

define noundef nonnull ptr @lookup(i32 noundef %s) {
entry:
  %0 = icmp ult i32 %s, 7
  br i1 %0, label %table, label %return

table:
  %gep = getelementptr inbounds [7 x ptr], ptr @.table, i32 0, i32 %s
  %element = load ptr, ptr %gep, align 4
; ABS:    %element = load ptr, ptr %gep, align 4
; REL:    {{%.*}} = call ptr @llvm.load.relative.i32(ptr @reltable.lookup, i32 {{%.*}})
  br label %return

return:                                           ; preds = %entry, %switch.lookup
  %ret = phi ptr [ %element, %table ], [ @.str.default, %entry ]
  ret ptr %ret
}
