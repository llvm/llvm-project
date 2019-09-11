
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @referencedbyglobal() {
    ret void
}

define internal void @localreferencedbyglobal() {
    ret void
}

@someglobal = internal unnamed_addr constant i8* bitcast (void ()* @referencedbyglobal to i8*)
@someglobal2 = internal unnamed_addr constant i8* bitcast (void ()* @localreferencedbyglobal to i8*)
@ptr = global i8** null
@ptr2 = global i8** null

define  void @bar() #0 align 2 {
  store i8** getelementptr inbounds (i8*, i8** @someglobal, i64 0) , i8*** @ptr, align 8
  store i8** getelementptr inbounds (i8*, i8** @someglobal2, i64 0) , i8*** @ptr2, align 8
  ret void
}
