; The only pass criterion is that spirv-val considers output valid.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

%subgr = type { i64, i64 }
%t_range = type { %t_arr }
%t_arr = type { [1 x i64] }
%t_arr2 = type { [4 x i32] }

define internal spir_func noundef i32 @geti32() {
entry:
  ret i32 100
}

define internal spir_func noundef i64 @geti64() {
entry:
  ret i64 200
}

define internal spir_func void @enable_if(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, i64 noundef %dim0) {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %dim0.addr = alloca i64, align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  store i64 %dim0, ptr %dim0.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %0 = load i64, ptr %dim0.addr, align 8
  call spir_func void @enable_if_2(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this1, i64 noundef %0)
  ret void
}


define internal spir_func void @test(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %this, ptr addrspace(4) noundef align 4 dereferenceable(16) %bits, ptr noundef byval(%t_range) align 8 %pos) {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %bits.addr = alloca ptr addrspace(4), align 8
  %cur_pos = alloca i64, align 8
  %__range4 = alloca ptr addrspace(4), align 8
  %__begin0 = alloca ptr addrspace(4), align 8
  %__end0 = alloca ptr addrspace(4), align 8
  %cleanup.dest.slot = alloca i32, align 4
  %elem = alloca ptr addrspace(4), align 8
  %agg.tmp = alloca %t_range, align 8
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  store ptr addrspace(4) %bits, ptr %bits.addr, align 8
  %pos.ascast = addrspacecast ptr %pos to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %call = call spir_func noundef i64 @getp(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %pos.ascast, i32 noundef 0)
  store i64 %call, ptr %cur_pos, align 8
  %0 = load ptr addrspace(4), ptr %bits.addr, align 8
  store ptr addrspace(4) %0, ptr %__range4, align 8
  %1 = load ptr addrspace(4), ptr %__range4, align 8
  %call2 = call spir_func noundef ptr addrspace(4) @beginp(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %1)
  store ptr addrspace(4) %call2, ptr %__begin0, align 8
  %2 = load ptr addrspace(4), ptr %__range4, align 8
  %call3 = call spir_func noundef ptr addrspace(4) @endp(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %2)
  store ptr addrspace(4) %call3, ptr %__end0, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load ptr addrspace(4), ptr %__begin0, align 8
  %4 = load ptr addrspace(4), ptr %__end0, align 8
  %cmp = icmp ne ptr addrspace(4) %3, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  %5 = load ptr addrspace(4), ptr %__begin0, align 8
  store ptr addrspace(4) %5, ptr %elem, align 8
  %6 = load i64, ptr %cur_pos, align 8
  %call4 = call spir_func noundef i32 @maskp(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %this1)
  %conv = zext i32 %call4 to i64
  %cmp5 = icmp ult i64 %6, %conv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %7 = load ptr addrspace(4), ptr %elem, align 8
  %8 = load i64, ptr %cur_pos, align 8
  call spir_func void @enable_if(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %agg.tmp.ascast, i64 noundef %8)
  call spir_func void @extract_bits(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %this1, ptr addrspace(4) noundef align 4 dereferenceable(4) %7, ptr noundef byval(%t_range) align 8 %agg.tmp)
  %9 = load i64, ptr %cur_pos, align 8
  %add = add i64 %9, 32
  store i64 %add, ptr %cur_pos, align 8
  br label %if.end

if.else:                                          ; preds = %for.body
  %10 = load ptr addrspace(4), ptr %elem, align 8
  store i32 0, ptr addrspace(4) %10, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %11 = load ptr addrspace(4), ptr %__begin0, align 8
  %incdec.ptr = getelementptr inbounds nuw i32, ptr addrspace(4) %11, i32 1
  store ptr addrspace(4) %incdec.ptr, ptr %__begin0, align 8
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

define internal spir_func noundef i64 @getp(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, i32 noundef %dimension) {
entry:
  %this.addr.i = alloca ptr addrspace(4), align 8
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64, align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %dimension.addr = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  store i32 %dimension, ptr %dimension.addr, align 4
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %0 = load i32, ptr %dimension.addr, align 4
  store ptr addrspace(4) %this1, ptr %this.addr.i, align 8
  store i32 %0, ptr %dimension.addr.i, align 4
  %this1.i = load ptr addrspace(4), ptr %this.addr.i, align 8
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %1 = load i32, ptr %dimension.addr, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], ptr addrspace(4) %common_array1, i64 0, i64 %idxprom
  %2 = load i64, ptr addrspace(4) %arrayidx, align 8
  ret i64 %2
}

define internal spir_func noundef ptr addrspace(4) @beginp(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this) {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %MData1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %arraydecay2 = bitcast ptr addrspace(4) %MData1 to ptr addrspace(4)
  ret ptr addrspace(4) %arraydecay2
}

define internal spir_func noundef ptr addrspace(4) @endp(ptr addrspace(4) noundef align 4 dereferenceable_or_null(16) %this) {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %MData1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %arraydecay2 = bitcast ptr addrspace(4) %MData1 to ptr addrspace(4)
  %add.ptr = getelementptr inbounds nuw i32, ptr addrspace(4) %arraydecay2, i64 4
  ret ptr addrspace(4) %add.ptr
}

define internal spir_func noundef i32 @maskp(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %this) {
entry:
  %retval = alloca i32, align 4
  %this.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %bits_num = getelementptr inbounds nuw %subgr, ptr addrspace(4) %this1, i32 0, i32 1
  %0 = load i64, ptr addrspace(4) %bits_num, align 8
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

define internal spir_func void @enable_if_2(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this, i64 noundef %dim0) {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %dim0.addr = alloca i64, align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  store i64 %dim0, ptr %dim0.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %common_array1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr %dim0.addr, align 8
  store i64 %0, ptr addrspace(4) %common_array1, align 8
  ret void
}

define internal spir_func void @extract_bits(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %this, ptr addrspace(4) noundef align 4 dereferenceable(4) %bits, ptr noundef byval(%t_range) align 8 %pos) {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %bits.addr = alloca ptr addrspace(4), align 8
  %Res = alloca i64, align 8
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  store ptr addrspace(4) %bits, ptr %bits.addr, align 8
  %pos.ascast = addrspacecast ptr %pos to ptr addrspace(4)
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %Bits1 = bitcast ptr addrspace(4) %this1 to ptr addrspace(4)
  %0 = load i64, ptr addrspace(4) %Bits1, align 8
  store i64 %0, ptr %Res, align 8
  %bits_num = getelementptr inbounds nuw %subgr, ptr addrspace(4) %this1, i32 0, i32 1
  %1 = load i64, ptr addrspace(4) %bits_num, align 8
  %call = call spir_func noundef i64 @geti64()
  %2 = load i64, ptr %Res, align 8
  %and = and i64 %2, %call
  store i64 %and, ptr %Res, align 8
  %call2 = call spir_func noundef i64 @geti64()
  %call3 = call spir_func noundef i32 @geti32()
  %conv = zext i32 %call3 to i64
  %cmp = icmp ult i64 %call2, %conv
  br i1 %cmp, label %if.then, label %if.else

if.else:                                          ; preds = %entry
  %3 = load ptr addrspace(4), ptr %bits.addr, align 8
  store i32 0, ptr addrspace(4) %3, align 4
  br label %if.end11

if.then:                                          ; preds = %entry
  %call4 = call spir_func noundef i64 @geti64()
  %cmp5 = icmp ugt i64 %call4, 0
  br i1 %cmp5, label %if.then6, label %if.end

if.then6:                                         ; preds = %if.then
  %call7 = call spir_func noundef i64 @geti64()
  %4 = load i64, ptr %Res, align 8
  %shr = lshr i64 %4, %call7
  store i64 %shr, ptr %Res, align 8
  br label %if.end

if.end:                                           ; preds = %if.then6, %if.then
  %call8 = call spir_func noundef i64 @geti64()
  %5 = load i64, ptr %Res, align 8
  %and9 = and i64 %5, %call8
  store i64 %and9, ptr %Res, align 8
  %6 = load i64, ptr %Res, align 8
  %conv10 = trunc i64 %6 to i32
  %7 = load ptr addrspace(4), ptr %bits.addr, align 8
  store i32 %conv10, ptr addrspace(4) %7, align 4
  br label %if.end11

if.end11:                                         ; preds = %if.else, %if.end
  ret void
}
