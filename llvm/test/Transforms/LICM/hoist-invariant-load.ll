; REQUIRES: asserts
; RUN: opt < %s -licm -verify-memoryssa -disable-basic-aa -stats -S 2>&1 | grep "1 licm"

@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"foo\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global ptr @"\01L_OBJC_METH_VAR_NAME_", section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [3 x ptr] [ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_SELECTOR_REFERENCES_", ptr @"\01L_OBJC_IMAGE_INFO"], section "llvm.metadata"

define void @test(ptr %x) uwtable ssp {
entry:
  %x.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  store ptr %x, ptr %x.addr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp ult i32 %0, 10000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load ptr, ptr %x.addr, align 8
  %2 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", !invariant.load !0
  %call = call ptr @objc_msgSend(ptr %1, ptr %2)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare ptr @objc_msgSend(ptr, ptr, ...) nonlazybind

!0 = !{}
