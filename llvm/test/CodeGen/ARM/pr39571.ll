; RUN: llc < %s -mtriple armv4t-unknown-linux-gnueabi -mattr=+strict-align

; Avoid crash from forwarding indexed-loads back to store.
%struct.anon = type { ptr, %struct.mb }
%struct.ma = type { i8 }
%struct.mb = type { i8, i8 }
%struct.anon.0 = type { %struct.anon.1 }
%struct.anon.1 = type { %struct.ds }
%struct.ds = type <{ i8, %union.ie }>
%union.ie = type { %struct.ib }
%struct.ib = type { i8, i8, i16 }

@a = common dso_local local_unnamed_addr global ptr null, align 4
@b = common dso_local local_unnamed_addr global %struct.anon.0 zeroinitializer, align 1

; Function Attrs: norecurse nounwind
define dso_local void @func() local_unnamed_addr {
entry:
  %0 = load ptr, ptr @a, align 4
  %1 = load ptr, ptr %0, align 4
  %c.sroa.0.0.copyload = load i8, ptr %1, align 1
  %cb = getelementptr inbounds %struct.anon, ptr %0, i32 0, i32 1
  %band = getelementptr inbounds %struct.anon, ptr %0, i32 0, i32 1, i32 1
  store i8 %c.sroa.0.0.copyload, ptr %band, align 4
  store i8 6, ptr getelementptr inbounds (%struct.anon.0, ptr @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0), align 1
  store i8 2, ptr getelementptr inbounds (%struct.anon.0, ptr @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1), align 1
  %2 = load i32, ptr getelementptr inbounds (%struct.anon.0, ptr @b, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0), align 1
  store i32 %2, ptr %cb, align 1
  ret void
}
