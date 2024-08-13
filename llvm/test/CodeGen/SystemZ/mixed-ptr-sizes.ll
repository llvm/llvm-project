; RUN: llc < %s -mtriple s390x-ibm-zos | FileCheck %s
; Source to regenerate:
; struct Foo {
;   int * __ptr32 p32;
;   int *p64;
;   char *cp64;
; };
; void use_foo(Foo *f);
;
; // Assiging a ptr32 value to a 64-bit pointer
; void ptr32_to_ptr(Foo *f, int * __ptr32 i) {
;   f->p64 = i;
;   use_foo(f);
; }
;
; // Assigning a 64-bit ptr value to a ptr32
; void ptr_to_ptr32(Foo *f, int *i) {
;   f->p32 = i;
;   use_foo(f);
; }
;
; // Assigning a ptr32 value to a ptr32 value
; void ptr32_to_ptr32(Foo *f, int * __ptr32 i) {
;   f->p32 = i;
;   use_foo(f);
; }
;
; void ptr_to_ptr(Foo *f, int *i) {
;   f->p64 = i;
;   use_foo(f);
; }
;
; void test_indexing(Foo *f) {
;  f->cp64 = ((char * __ptr32 *)1028)[1];
;  use_foo(f);
; }
;
; void test_indexing_2(Foo *f) {
;   f->cp64 = ((char *** __ptr32 *)1028)[1][2][3];
;   use_foo(f);
; }
;
; unsigned long* test_misc() {
;   unsigned long* x = (unsigned long*)((char***** __ptr32*)1208)[0][11][1][113][149];
;   return x;
; }
;
; char* __ptr32* __ptr32 test_misc_2() {
;   static char* __ptr32* __ptr32 res = 0;
;   if (res == 0) {
;     res = ((char* __ptr32* __ptr32* __ptr32* __ptr32*)0)[4][136][6];
;   }
;   return res;
; }
;
; unsigned short test_misc_3() {
;   unsigned short this_asid = ((unsigned short*)(*(char* __ptr32*)(0x224)))[18];
;   return this_asid;
; }
;
; int test_misc_4() {
;   int a = (*(int*)(80 + ((char**** __ptr32*)1208)[0][11][1][123]) > 0x040202FF);
;   return a;
; }
;
; void test_misc_5(struct Foo *f) {
;   f->cp64  = *(char* __ptr32 *)(PSA_PTR + PSAAOLD);
;   use_foo(f);
; }
;
; int get_processor_count() {
;  return ((char * __ptr32 * __ptr32 *)0)[4][165][53];
; }
;
; void spill_ptr32_args_to_registers( char *__ptr32 p ) {
;   void g ( int, ... );
;   g ( 5, p, p, p, p, p );
; }
;
; $ clang -cc1 -triple s390x-ibm-zos -fzos-extensions -O2 -S t.cpp
;
; For the last test case:
;
;#include <stdlib.h>
;
;int foo();
;
;typedef struct qa_area {/* Area descriptor                */
;  char* __ptr32 text;           /* Start address of area          */
;  int length;      /* Size of area in bytes          */
;} qa_area;
;
;int main() {
;  qa_area* __ptr32 fap_asm_option_a = (qa_area*)__malloc31(sizeof(qa_area));
;
;  //((qa_area*)fap_asm_option_a)->length   = foo(); //PASSES
;  fap_asm_option_a->length = foo();                 //CRASHES
;  return 0;
;}

%struct.Foo = type { i32 addrspace(1)*, i32*, i8* }
declare dso_local void @use_foo(%struct.Foo*)

define void @ptr32_to_ptr(%struct.Foo* %f, i32 addrspace(1)* %i) {
; CHECK-LABEL: ptr32_to_ptr:
; CHECK:       llgtr 0, 2
; CHECK-NEXT:  stg   0, 8(1)
entry:
  %0 = addrspacecast i32 addrspace(1)* %i to i32*
  %p64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %0, i32** %p64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @ptr_to_ptr32(%struct.Foo* %f, i32* %i) {
entry:
; CHECK-LABEL: ptr_to_ptr32:
; CHECK:       nilh 2, 32767
; CHECK-NEXT:  st   2, 0(1)
  %0 = addrspacecast i32* %i to i32 addrspace(1)*
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32 addrspace(1)* %0, i32 addrspace(1)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @ptr32_to_ptr32(%struct.Foo* %f, i32 addrspace(1)* %i) {
entry:
; CHECK-LABEL: ptr32_to_ptr32:
; CHECK:       st 2, 0(1)
  %p32 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32 addrspace(1)* %i, i32 addrspace(1)** %p32, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @ptr_to_ptr(%struct.Foo* %f, i32* %i) {
; CHECK-LABEL: ptr_to_ptr:
; CHECK:       stg 2, 8(1)
  %p64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %i, i32** %p64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @test_indexing(%struct.Foo* %f) {
entry:
; CHECK-LABEL: test_indexing:
; CHECK:       l     0, 1032
; CHECK:       llgtr 0, 0
; CHECK:       stg   0, 16(1)
  %0 = load i8 addrspace(1)*, i8 addrspace(1)** inttoptr (i64 1032 to i8 addrspace(1)**), align 8
  %1 = addrspacecast i8 addrspace(1)* %0 to i8*
  %cp64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 2
  store i8* %1, i8** %cp64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define void @test_indexing_2(%struct.Foo* %f) {
entry:
; CHECK-LABEL: test_indexing_2:
; CHECK:       lhi   0, 16
; CHECK-NEXT:  a     0, 1032
; CHECK-NEXT:  llgtr 2, 0
; CHECK:       lg    0, 24(2)
; CHECK:       stg   0, 16(1)
  %0 = load i8** addrspace(1)*, i8** addrspace(1)** inttoptr (i64 1032 to i8** addrspace(1)**), align 8
  %arrayidx = getelementptr inbounds i8**, i8** addrspace(1)* %0, i32 2
  %1 = load i8**, i8** addrspace(1)* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i8*, i8** %1, i64 3
  %2 = bitcast i8** %arrayidx1 to i64*
  %3 = load i64, i64* %2, align 8
  %cp64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 2
  %4 = bitcast i8** %cp64 to i64*
  store i64 %3, i64* %4, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define i64* @test_misc() {
entry:
; CHECK-LABEL: test_misc:
; CHECK:       lhi   0, 88
; CHECK-NEXT:  a     0, 1208
; CHECK-NEXT:  llgtr 1, 0
; CHECK-NEXT:  lg    1, 0(1)
; CHECK-NEXT:  lg    1, 8(1)
; CHECK-NEXT:  lg    1, 904(1)
; CHECK-NEXT:  lg    3, 1192(1)
  %0 = load i8**** addrspace(1)*, i8**** addrspace(1)** inttoptr (i64 1208 to i8**** addrspace(1)**), align 8
  %arrayidx = getelementptr inbounds i8****, i8**** addrspace(1)* %0, i32 11
  %1 = load i8****, i8**** addrspace(1)* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i8***, i8**** %1, i64 1
  %2 = load i8***, i8**** %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds i8**, i8*** %2, i64 113
  %3 = load i8**, i8*** %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds i8*, i8** %3, i64 149
  %4 = bitcast i8** %arrayidx3 to i64**
  %5 = load i64*, i64** %4, align 8
  ret i64* %5
}

define i8 addrspace(1)* addrspace(1)* @test_misc_2() {
entry:
; CHECK-LABEL: test_misc_2:
; CHECK:       lhi   0, 544
; CHECK:       a     0, 16
; CHECK:       llgtr 1, 0
; CHECK:       lhi   0, 24
; CHECK:       a     0, 0(1)
; CHECK:       llgtr 1, 0
  %0 = load i8 addrspace(1)* addrspace(1)* addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)* addrspace(1)* addrspace(1)** inttoptr (i64 16 to i8 addrspace(1)* addrspace(1)* addrspace(1)* addrspace(1)**), align 16
  %arrayidx = getelementptr inbounds i8 addrspace(1)* addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)* addrspace(1)* addrspace(1)* %0, i32 136
  %1 = load i8 addrspace(1)* addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)* addrspace(1)* addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i8 addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)* addrspace(1)* %1, i32 6
  %2 = load i8 addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)* addrspace(1)* %arrayidx1, align 4
  ret i8 addrspace(1)* addrspace(1)* %2
}

define zeroext i16 @test_misc_3() local_unnamed_addr #3 {
entry:
; CHECK-LABEL: test_misc_3:
; CHECK:       lhi   0, 36
; CHECK-NEXT:  a     0, 548
; CHECK-NEXT:  llgtr 1, 0
; CHECK-NEXT:  llgh  3, 0(1)
  %0 = load i16 addrspace(1)*, i16 addrspace(1)** inttoptr (i64 548 to i16 addrspace(1)**), align 4
  %arrayidx2 = getelementptr inbounds i16, i16 addrspace(1)* %0, i32 18
  %arrayidx = addrspacecast i16 addrspace(1)* %arrayidx2 to i16*
  %1 = load i16, i16* %arrayidx, align 2
  ret i16 %1
}

define signext i32 @test_misc_4() {
entry:
; CHECK-LABEL: test_misc_4:
; CHECK:       lhi   0, 88
; CHECK-NEXT:  a     0, 1208
; CHECK-NEXT:  llgtr 1, 0
; CHECK-NEXT:  lg    1, 0(1)
; CHECK-NEXT:  lg    1, 8(1)
; CHECK-NEXT:  lg    1, 984(1)
; CHECK-NEXT:  iilf  0, 67240703
; CHECK-NEXT:  c     0, 80(1)
  %0 = load i8*** addrspace(1)*, i8*** addrspace(1)** inttoptr (i64 1208 to i8*** addrspace(1)**), align 8
  %arrayidx = getelementptr inbounds i8***, i8*** addrspace(1)* %0, i32 11
  %1 = load i8***, i8*** addrspace(1)* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i8**, i8*** %1, i64 1
  %2 = load i8**, i8*** %arrayidx1, align 8
  %arrayidx2 = getelementptr inbounds i8*, i8** %2, i64 123
  %3 = load i8*, i8** %arrayidx2, align 8
  %add.ptr = getelementptr inbounds i8, i8* %3, i64 80
  %4 = bitcast i8* %add.ptr to i32*
  %5 = load i32, i32* %4, align 4
  %cmp = icmp sgt i32 %5, 67240703
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define void @test_misc_5(%struct.Foo* %f) {
entry:
; CHECK-LABEL: test_misc_5:
; CHECK:       l     0, 548
; CHECK-NEXT:  lg  6, 8(5)
; CHECK-NEXT:  lg  5, 0(5)
; CHECK-NEXT:  llgtr 0, 0
; CHECK-NEXT:  stg   0, 16(1)
  %0 = load i8 addrspace(1)*, i8 addrspace(1)** inttoptr (i64 548 to i8 addrspace(1)**), align 4
  %1 = addrspacecast i8 addrspace(1)* %0 to i8*
  %cp64 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 2
  store i8* %1, i8** %cp64, align 8
  tail call void @use_foo(%struct.Foo* %f)
  ret void
}

define signext i32 @get_processor_count() {
entry:
; CHECK-LABEL: get_processor_count:
; CHECK: lhi 0, 660
; CHECK-NEXT: a 0, 16
; CHECK-NEXT: llgtr 1, 0
; CHECK-NEXT: lhi 0, 53
; CHECK-NEXT: a 0, 0(1)
; CHECK-NEXT: llgtr 1, 0
; CHECK-NEXT: lgb 3, 0(1)
  %0 = load i8 addrspace(1)* addrspace(1)*, i8 addrspace(1)* addrspace(1)** inttoptr (i64 16 to i8 addrspace(1)* addrspace(1)**), align 16
  %arrayidx = getelementptr inbounds i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %0, i32 165
  %1 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %1, i32 53
  %2 = load i8, i8 addrspace(1)* %arrayidx1, align 1
  %conv = sext i8 %2 to i32
  ret i32 %conv
}

define void @spill_ptr32_args_to_registers(i8 addrspace(1)* %p) {
entry:
; CHECK-LABEL: spill_ptr32_args_to_registers:
; CHECK:         stmg 6, 7, 1872(4)
; CHECK-NEXT:    aghi 4, -192
; CHECK-NEXT:    lgr 2, 1
; CHECK-NEXT:    lg 6, 24(5)
; CHECK-NEXT:    lg 5, 16(5)
; CHECK-NEXT:    stg 1, 2216(4)
; CHECK-NEXT:    stg 1, 2208(4)
; CHECK-NEXT:    lghi 1, 5
; CHECK-NEXT:    stg 2, 2200(4)
; CHECK-NEXT:    lgr 3, 2
; CHECK-NEXT:    basr 7, 6
; CHECK-NEXT:    bcr 0, 0
; CHECK-NEXT:    lg 7, 2072(4)
; CHECK-NEXT:    aghi 4, 192
; CHECK-NEXT:    b 2(7)
  tail call void (i32, ...) @g(i32 noundef signext 5, i8 addrspace(1)* noundef %p, i8 addrspace(1)* noundef %p, i8 addrspace(1)* noundef %p, i8 addrspace(1)* noundef %p, i8 addrspace(1)* noundef %p)
   ret void
}
declare void @g(i32 noundef signext, ...)

; The resulting instructions may look odd on first view but it is a result of
; the C code. __malloc31() returns a 64 bit pointer, thus the sequence
;
;        la      1, 4(8)
;        llgtr   1, 1
;
; references the length attribute via the 64 bit pointer, and performs the
; cast to __ptr32, setting the upper 32 bit to zero.
;
define signext i32 @setlength() {
; CHECK-LABEL: setlength:
; CHECK: basr    7, 6
; CHECK: lgr     [[MALLOC:[0-9]+]], 3
; CHECK: basr    7, 6
; CHECK: lgr     [[LENGTH:[0-9]+]], 3
; CHECK: la      [[ADDR:[0-9]+]], 4([[MALLOC]])
; CHECK: llgtr   [[ADDR]], [[ADDR]]
; CHECK: stg     [[LENGTH]], 0([[ADDR]])
entry:
  %call = tail call i8* @__malloc31(i64 noundef 8)
  %call1 = tail call signext i32 bitcast (i32 (...)* @foo to i32 ()*)()
  %length = getelementptr inbounds i8, i8* %call, i64 4
  %0 = bitcast i8* %length to i32*
  %1 = addrspacecast i32* %0 to i32 addrspace(1)*
  store i32 %call1, i32 addrspace(1)* %1, align 4
  ret i32 0
}

; Same as test before, but this time calling
;  extern char* __ptr32 domalloc(unsigned long);
; instead of __malloc31(). Note the different instruction sequence, because
; the function now returns a __ptr32.
;
define hidden signext i32 @setlength2() {
; CHECK-LABEL: setlength2:
; CHECK: basr    7, 6
; CHECK: lgr     [[MALLOC:[0-9]+]], 3
; CHECK: basr    7, 6
; CHECK: lgr     [[LENGTH:[0-9]+]], 3
; CHECK: ahi     [[MALLOC]], 4
; CHECK: llgtr   [[ADDR]], [[MALLOC]]
; CHECK: stg     [[LENGTH]], 0([[ADDR]])
entry:
  %call = tail call i8 addrspace(1)* @domalloc(i64 noundef 8)
  %call1 = tail call signext i32 bitcast (i32 (...)* @foo to i32 ()*)() #2
  %length = getelementptr inbounds i8, i8 addrspace(1)* %call, i32 4
  %0 = bitcast i8 addrspace(1)* %length to i32 addrspace(1)*
  store i32 %call1, i32 addrspace(1)* %0, align 4
  ret i32 0
}

declare i8* @__malloc31(i64 noundef)

declare signext i32 @foo(...)

declare i8 addrspace(1)* @domalloc(i64 noundef)
