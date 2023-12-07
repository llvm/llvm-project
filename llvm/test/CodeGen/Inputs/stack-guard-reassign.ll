define i32 @fn(i8* %str) #0 {
entry:
  %str.addr = alloca i8*, align 4
  %buffer = alloca [65536 x i8], align 1
  store i8* %str, i8** %str.addr, align 4
  %arraydecay = getelementptr inbounds [65536 x i8], [65536 x i8]* %buffer, i32 0, i32 0
  %0 = load i8*, i8** %str.addr, align 4
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [65536 x i8], [65536 x i8]* %buffer, i32 0, i32 0
  %call2 = call i32 @puts(i8* %arraydecay1)
  %arrayidx = getelementptr inbounds [65536 x i8], [65536 x i8]* %buffer, i32 0, i32 65535
  %1 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %1 to i32
  ret i32 %conv
}

declare i8* @strcpy(i8*, i8*)

declare i32 @puts(i8*)

attributes #0 = { noinline nounwind optnone ssp }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"direct-access-external-data", i32 1}
