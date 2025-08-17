define i32 @fn(ptr %str) #0 {
entry:
  %str.addr = alloca ptr, align 4
  %buffer = alloca [65536 x i8], align 1
  store ptr %str, ptr %str.addr, align 4
  %arraydecay = getelementptr inbounds [65536 x i8], ptr %buffer, i32 0, i32 0
  %0 = load ptr, ptr %str.addr, align 4
  %call = call ptr @strcpy(ptr %arraydecay, ptr %0)
  %arraydecay1 = getelementptr inbounds [65536 x i8], ptr %buffer, i32 0, i32 0
  %call2 = call i32 @puts(ptr %arraydecay1)
  %arrayidx = getelementptr inbounds [65536 x i8], ptr %buffer, i32 0, i32 65535
  %1 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %1 to i32
  ret i32 %conv
}

declare ptr @strcpy(ptr, ptr)

declare i32 @puts(ptr)

attributes #0 = { noinline nounwind optnone ssp }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"direct-access-external-data", i32 1}
