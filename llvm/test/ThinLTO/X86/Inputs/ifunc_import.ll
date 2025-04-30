target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define dso_local void @called() {
  ret void
}

@globalIfunc = ifunc void (), bitcast (void ()* ()* @globalResolver to void ()*)
@globalWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @globalResolver to void ()*)
@globalLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @globalResolver to void ()*)
@globalWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @globalResolver to void ()*)
@globalLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @globalResolver to void ()*)
define hidden void ()* @globalResolver() {
entry:
  ret void ()* @called
}

@internalIfunc = ifunc void (), bitcast (void ()* ()* @internalResolver to void ()*)
@internalWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @internalResolver to void ()*)
@internalLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @internalResolver to void ()*)
@internalWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @internalResolver to void ()*)
@internalLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @internalResolver to void ()*)
define internal void ()* @internalResolver() {
entry:
  ret void ()* @called
}

@linkonceODRIfunc = ifunc void (), bitcast (void ()* ()* @linkonceODRResolver to void ()*)
@linkonceODRWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @linkonceODRResolver to void ()*)
@linkonceODRLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @linkonceODRResolver to void ()*)
@linkonceODRWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @linkonceODRResolver to void ()*)
@linkonceODRLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @linkonceODRResolver to void ()*)
define linkonce_odr void ()* @linkonceODRResolver() {
entry:
  ret void ()* @called
}

@weakODRIfunc = ifunc void (), bitcast (void ()* ()* @weakODRResolver to void ()*)
@weakODRWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @weakODRResolver to void ()*)
@weakODRLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @weakODRResolver to void ()*)
@weakODRWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @weakODRResolver to void ()*)
@weakODRLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @weakODRResolver to void ()*)
define weak_odr void ()* @weakODRResolver() {
entry:
  ret void ()* @called
}

@linkonceIfunc = ifunc void (), bitcast (void ()* ()* @linkonceResolver to void ()*)
@linkonceWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @linkonceResolver to void ()*)
@linkonceLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @linkonceResolver to void ()*)
@linkonceWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @linkonceResolver to void ()*)
@linkonceLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @linkonceResolver to void ()*)
define linkonce void ()* @linkonceResolver() {
entry:
  ret void ()* @called
}

@weakIfunc = ifunc void (), bitcast (void ()* ()* @weakResolver to void ()*)
@weakWeakIfunc = weak ifunc void (), bitcast (void ()* ()* @weakResolver to void ()*)
@weakLinkonceIfunc = linkonce ifunc void (), bitcast (void ()* ()* @weakResolver to void ()*)
@weakWeakODRIfunc = weak_odr ifunc void (), bitcast (void ()* ()* @weakResolver to void ()*)
@weakLinkonceODRIfunc = linkonce_odr ifunc void (), bitcast (void ()* ()* @weakResolver to void ()*)
define weak void ()* @weakResolver() {
entry:
  ret void ()* @called
}
