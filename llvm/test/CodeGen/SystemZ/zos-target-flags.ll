; RUN: llc -mtriple=s390x-ibm-zos -stop-after=systemz-isel --simplify-mir < %s | FileCheck %s


declare i64 @calc(i64 noundef, ptr noundef)
declare i64 @morework(i64 noundef)

@i = external local_unnamed_addr global i64, align 8

define i64 @work() {
entry:
; CHECK:    %{{.*}}:addr64bit = ADA_ENTRY_VALUE target-flags(systemz-ada-datasymboladdr) @i,
; CHECK:    %{{.*}}:addr64bit = ADA_ENTRY_VALUE target-flags(systemz-ada-directfuncdesc) @calc,
; CHECK:    %{{.*}}:addr64bit = ADA_ENTRY_VALUE target-flags(systemz-ada-indirectfuncdesc) @morework,
  %0 = load i64, ptr @i, align 8
  %call = tail call i64 @calc(i64 noundef %0, ptr noundef nonnull @morework) #2
  ret i64 %call
}
