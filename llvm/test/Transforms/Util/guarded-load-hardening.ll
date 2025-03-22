; RUN: opt -S -passes=guarded-load-hardening -guarded-load-hardening < %s | FileCheck %s --check-prefix ON
; RUN: opt -S -passes=guarded-load-hardening < %s | FileCheck %s --check-prefix OFF

; If the feature isn't enabled, we shouldn't see the intrinsic generated.
; OFF-NOT:  call void @llvm.speculative.data.barrier()

; Scenario: From the MSVC blog post
; https://devblogs.microsoft.com/cppblog/spectre-mitigations-in-msvc/
; From the C++:
; int guarded_index_load_from_array(unsigned char* indexes, unsigned char* data, int index, int indexes_len) {
;     if (index < indexes_len) {
;         unsigned char sub_index = indexes[index];
;         return data[sub_index * 64];
;     }
;     return 0;
; }
define dso_local noundef i32 @guarded_index_load_from_array(
  ptr nocapture noundef readonly %indexes,
  ptr nocapture noundef readonly %data,
  i32 noundef %index,
  i32 noundef %indexes_len) {
entry:
  %cmp = icmp slt i32 %index, %indexes_len
  br i1 %cmp, label %if.then, label %return

; ON-LABEL: define dso_local noundef i32 @guarded_index_load_from_array
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %idxprom = sext i32 %index to i64
  %arrayidx = getelementptr inbounds i8, ptr %indexes, i64 %idxprom
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i64
  %mul = shl nuw nsw i64 %conv, 6
  %arrayidx2 = getelementptr inbounds i8, ptr %data, i64 %mul
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  br label %return

return:
  %retval.0 = phi i32 [ %conv3, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

; Scenario: As above (the MSVC blog post), but with an indirect call.
; From the C++:
; using FPtr = int(*)();
; int guarded_fptr_call_from_array(FPtr* funcs, int index, int funcs_len) {
;     if (index < funcs_len) {
;         return funcs[index * 4]();
;     }
;     return 0;
; }
define dso_local noundef i32 @guarded_fptr_call_from_array(
  ptr nocapture noundef readonly %funcs,
  i32 noundef %index,
  i32 noundef %funcs_len) local_unnamed_addr {
entry:
  %cmp = icmp slt i32 %index, %funcs_len
  br i1 %cmp, label %if.then, label %return

; ON-LABEL: define dso_local noundef i32 @guarded_fptr_call_from_array
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %mul = shl nsw i32 %index, 2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds ptr, ptr %funcs, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  %call = tail call noundef i32 %0()
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

@temp = dso_local local_unnamed_addr global i8 0, align 1
@array1_size = external local_unnamed_addr global i32, align 4
@array2 = external local_unnamed_addr global [0 x i8], align 1
@array1 = external local_unnamed_addr global [0 x i8], align 1

; Scenario: As written in the Spectre paper
; From the C++:
; void victim_function(size_t x) {
;   if (x < array1_size) {
;     temp &= array2[array1[x] * 512];
;   }
; }
define dso_local void @victim_function(i64 noundef %x) local_unnamed_addr {
entry:
  %0 = load i32, ptr @array1_size, align 4
  %conv = zext i32 %0 to i64
  %cmp = icmp ult i64 %x, %conv
  br i1 %cmp, label %if.then, label %if.end

; ON-LABEL: define dso_local void @victim_function
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %arrayidx = getelementptr inbounds nuw [0 x i8], ptr @array1, i64 0, i64 %x
  %1 = load i8, ptr %arrayidx, align 1
  %conv1 = zext i8 %1 to i64
  %mul = shl nuw nsw i64 %conv1, 9
  %arrayidx2 = getelementptr inbounds [0 x i8], ptr @array2, i64 0, i64 %mul
  %2 = load i8, ptr %arrayidx2, align 1
  %3 = load i8, ptr @temp, align 1
  %and7 = and i8 %3, %2
  store i8 %and7, ptr @temp, align 1
  br label %if.end

if.end:
  ret void
}

; Scenario: Shift/multiply the index
; From the C++:
; void victim_function_alt03(size_t x) {
;   if (x < array1_size)
;     temp &= array2[array1[x << 1] * 512];
; }
define dso_local void @victim_function_alt03(i64 noundef %x) local_unnamed_addr {
entry:
  %0 = load i32, ptr @array1_size, align 4
  %conv = zext i32 %0 to i64
  %cmp = icmp ult i64 %x, %conv
  br i1 %cmp, label %if.then, label %if.end

; ON-LABEL: define dso_local void @victim_function_alt03
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %shl = shl nuw nsw i64 %x, 1
  %arrayidx = getelementptr inbounds nuw [0 x i8], ptr @array1, i64 0, i64 %shl
  %1 = load i8, ptr %arrayidx, align 1
  %conv1 = zext i8 %1 to i64
  %mul = shl nuw nsw i64 %conv1, 9
  %arrayidx2 = getelementptr inbounds [0 x i8], ptr @array2, i64 0, i64 %mul
  %2 = load i8, ptr %arrayidx2, align 1
  %3 = load i8, ptr @temp, align 1
  %and7 = and i8 %3, %2
  store i8 %and7, ptr @temp, align 1
  br label %if.end

if.end:
  ret void
}

; Scenario: Pointer arithmetic + memcmp
; From the C++:
; void victim_function_alt10(size_t x) {
;   if (x < array1_size)
;     temp = memcmp(&temp, array2+(array1[x] * 512), 1);
; }
define dso_local void @victim_function_alt10(i64 noundef %x) local_unnamed_addr {
entry:
  %0 = load i32, ptr @array1_size, align 4
  %conv = zext i32 %0 to i64
  %cmp = icmp ult i64 %x, %conv
  br i1 %cmp, label %if.then, label %if.end

; ON-LABEL: define dso_local void @victim_function_alt10
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %arrayidx = getelementptr inbounds nuw [0 x i8], ptr @array1, i64 0, i64 %x
  %1 = load i8, ptr %arrayidx, align 1
  %conv1 = zext i8 %1 to i64
  %mul = shl nuw nsw i64 %conv1, 9
  %add.ptr = getelementptr inbounds i8, ptr @array2, i64 %mul
  %lhsc = load i8, ptr @temp, align 1
  %rhsc = load i8, ptr %add.ptr, align 1
  %chardiff = sub i8 %lhsc, %rhsc
  store i8 %chardiff, ptr @temp, align 1
  br label %if.end

if.end:
  ret void
}

; Scenario: Index uses sum of two args
; From the C++:
; void victim_function_alt11(size_t x, size_t y) {
;   if ((x+y) < array1_size)
;     temp &= array2[array1[x+y] * 512];
; }
define dso_local void @victim_function_alt11(i64 noundef %x, i64 noundef %y) local_unnamed_addr {
entry:
  %add = add i64 %y, %x
  %0 = load i32, ptr @array1_size, align 4
  %conv = zext i32 %0 to i64
  %cmp = icmp ult i64 %add, %conv
  br i1 %cmp, label %if.then, label %if.end

; ON-LABEL: define dso_local void @victim_function_alt11
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %arrayidx = getelementptr inbounds nuw [0 x i8], ptr @array1, i64 0, i64 %add
  %1 = load i8, ptr %arrayidx, align 1
  %conv2 = zext i8 %1 to i64
  %mul = shl nuw nsw i64 %conv2, 9
  %arrayidx3 = getelementptr inbounds [0 x i8], ptr @array2, i64 0, i64 %mul
  %2 = load i8, ptr %arrayidx3, align 1
  %3 = load i8, ptr @temp, align 1
  %and9 = and i8 %3, %2
  store i8 %and9, ptr @temp, align 1
  br label %if.end

if.end:
  ret void
}

; Scenario: Invert the bits of the index
; From the C++:
; void victim_function_alt13(size_t x) {
;   if (x < array1_size)
;     temp &= array2[array1[x ^ 255] * 512];
; }
define dso_local void @victim_function_alt13(i64 noundef %x) local_unnamed_addr {
entry:
  %0 = load i32, ptr @array1_size, align 4
  %conv = zext i32 %0 to i64
  %cmp = icmp ult i64 %x, %conv
  br i1 %cmp, label %if.then, label %if.end

; ON-LABEL: define dso_local void @victim_function_alt13
; ON:       if.then:
; ON-NEXT:  call void @llvm.speculative.data.barrier()
if.then:
  %xor = xor i64 %x, 255
  %arrayidx = getelementptr inbounds nuw [0 x i8], ptr @array1, i64 0, i64 %xor
  %1 = load i8, ptr %arrayidx, align 1
  %conv1 = zext i8 %1 to i64
  %mul = shl nuw nsw i64 %conv1, 9
  %arrayidx2 = getelementptr inbounds [0 x i8], ptr @array2, i64 0, i64 %mul
  %2 = load i8, ptr %arrayidx2, align 1
  %3 = load i8, ptr @temp, align 1
  %and7 = and i8 %3, %2
  store i8 %and7, ptr @temp, align 1
  br label %if.end

if.end:
  ret void
}