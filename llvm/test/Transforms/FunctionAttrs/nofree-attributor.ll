; RUN: opt -function-attrs --disable-nofree-inference=false -S < %s | FileCheck %s --check-prefix=FNATTR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "nofree" function attribute.
; We use FIXME's to indicate problems and missing attributes.

; Free functions
declare void @free(i8* nocapture) local_unnamed_addr #1
declare noalias i8* @realloc(i8* nocapture, i64) local_unnamed_addr #0
declare void @_ZdaPv(i8*) local_unnamed_addr #2


; TEST 1 (positive case)
; FNATTR: Function Attrs: noinline norecurse nounwind readnone uwtable
; FNATTR-NEXT: define void @only_return()
define void @only_return() #0 {
    ret void
}


; TEST 2 (negative case)
; Only free
; void only_free(char* p) {
;    free(p);
; }

; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define void @only_free(i8* nocapture %0) local_unnamed_addr
define void @only_free(i8* nocapture %0) local_unnamed_addr #0 {
    tail call void @free(i8* %0) #1
    ret void
}


; TEST 3 (negative case)
; Free occurs in same scc.
; void free_in_scc1(char*p){
;    free_in_scc2(p);
; }
; void free_in_scc2(char*p){
;    free_in_scc1(p);
;    free(p);
; }


; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define void @free_in_scc1(i8* nocapture %0) local_unnamed_addr
define void @free_in_scc1(i8* nocapture %0) local_unnamed_addr #0 {
  tail call void @free_in_scc2(i8* %0) #1
  ret void
}


; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define void @free_in_scc2(i8* nocapture %0) local_unnamed_addr
define void @free_in_scc2(i8* nocapture %0) local_unnamed_addr #0 {
  %cmp = icmp eq i8* %0, null
  br i1 %cmp, label %rec, label %call
call:
  tail call void @free(i8* %0) #1
  br label %end
rec:
  tail call void @free_in_scc1(i8* %0)
  br label %end
end:
  ret void
}


; TEST 4 (positive case)
; Free doesn't occur.
; void mutual_recursion1(){
;    mutual_recursion2();
; }
; void mutual_recursion2(){
;     mutual_recursion1();
; }


; FNATTR: Function Attrs: noinline nounwind readnone uwtable
; FNATTR-NEXT: define void @mutual_recursion1()
define void @mutual_recursion1() #0 {
  call void @mutual_recursion2()
  ret void
}

; FNATTR: Function Attrs: noinline nounwind readnone uwtable
; FNATTR-NEXT: define void @mutual_recursion2()
define void @mutual_recursion2() #0 {
  call void @mutual_recursion1()
  ret void
}


; TEST 5
; C++ delete operation (negative case)
; void delete_op (char p[]){
;     delete [] p;
; }

; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define void @_Z9delete_opPc(i8* %0) local_unnamed_addr
define void @_Z9delete_opPc(i8* %0) local_unnamed_addr #0 {
  %2 = icmp eq i8* %0, null
  br i1 %2, label %4, label %3

; <label>:3:                                      ; preds = %1
  tail call void @_ZdaPv(i8* nonnull %0) #2
  br label %4

; <label>:4:                                      ; preds = %3, %1
  ret void
}


; TEST 6 (negative case)
; Call realloc
; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define noalias i8* @call_realloc(i8* nocapture %0, i64 %1) local_unnamed_addr
define noalias i8* @call_realloc(i8* nocapture %0, i64 %1) local_unnamed_addr #0 {
    %ret = tail call i8* @realloc(i8* %0, i64 %1) #2
    ret i8* %ret
}


; TEST 7 (positive case)
; Call function declaration with "nofree"


; FNATTR: Function Attrs: nofree noinline nounwind readnone uwtable
; FNATTR-NEXT: declare void @nofree_function()
declare void @nofree_function() nofree readnone #0

; FNATTR: Function Attrs: noinline nounwind readnone uwtable
; FNATTR-NEXT: define void @call_nofree_function()
define void @call_nofree_function() #0 {
    tail call void @nofree_function()
    ret void
}

; TEST 8 (negative case)
; Call function declaration without "nofree"


declare void @maybe_free() #0


; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR: define void @call_maybe_free()
define void @call_maybe_free() #0 {
    tail call void @maybe_free()
    ret void
}


; TEST 9 (negative case)
; Call both of above functions

; FNATTR: Function Attrs: noinline nounwind uwtable
; FNATTR-NEXT: define void @call_both()
define void @call_both() #0 {
    tail call void @maybe_free()
    tail call void @nofree_function()
    ret void
}


; TEST 10 (positive case)
; Call intrinsic function
; FNATTRS: Function Attrs: noinline readnone speculatable
; FNATTRS-NEXT: declare float @llvm.floor.f32(float %0)
declare float @llvm.floor.f32(float)

; FNATTRS: Function Attrs: noinline nounwind uwtable
; FNATTRS-NEXT: define void @call_floor(float %a)
; FIXME: missing nofree

define void @call_floor(float %a) #0 {
    tail call float @llvm.floor.f32(float %a)
    ret void
}

; TEST 11 (positive case)
; Check propagation.

; FNATTRS: Function Attrs: noinline nounwind uwtable
; FNATTRS-NEXT: define void @f1()
define void @f1() #0 {
    tail call void @nofree_function()
    ret void
}

; FNATTRS: Function Attrs: noinline nounwind uwtable
; FNATTRS-NEXT: define void @f2()
define void @f2() #0 {
    tail call void @f1()
    ret void
}


declare noalias i8* @malloc(i64)

attributes #0 = { nounwind uwtable noinline }
attributes #1 = { nounwind }
attributes #2 = { nobuiltin nounwind }
