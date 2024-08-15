; RUN: llc < %s -march=nvptx64 -mcpu=sm_20

define <6 x half> @half6() {
  ret <6 x half> zeroinitializer
}

define <10 x half> @half10() {
  ret <10 x half> zeroinitializer
}

define <14 x half> @half14() {
  ret <14 x half> zeroinitializer
}

define <18 x half> @half18() {
  ret <18 x half> zeroinitializer
}

define <998 x half> @half998() {
  ret <998 x half> zeroinitializer
}

define <12 x i8> @byte12() {
  ret <12 x i8> zeroinitializer
}

define <20 x i8> @byte20() {
  ret <20 x i8> zeroinitializer
}

define <24 x i8> @byte24() {
  ret <24 x i8> zeroinitializer
}

define <28 x i8> @byte28() {
  ret <28 x i8> zeroinitializer
}

define <36 x i8> @byte36() {
  ret <36 x i8> zeroinitializer
}

define <996 x i8> @byte996() {
  ret <996 x i8> zeroinitializer
}
