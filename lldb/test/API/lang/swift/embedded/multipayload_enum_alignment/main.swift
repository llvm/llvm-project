// Every size and offset this test checks is expressed in explicitly sized types
// (Int64/UInt8/Int8), so none of them depend on the pointer width. They do
// depend on Int64 being 8-byte aligned, which the test gates on.

// A 33-byte payload: four Int64s (alignment 8) plus a trailing UInt8. Its
// DW_AT_byte_size (33) is deliberately not a power of two and much larger than
// its real alignment (8).
struct Wide {
  var a: Int64 = 1
  var b: Int64 = 2
  var c: Int64 = 3
  var d: Int64 = 4
  var e: UInt8 = 5
}

// A multi-payload enum: emitted as a DW_TAG_structure_type whose first child is
// a DW_TAG_variant_part, with DW_AT_byte_size 34 and DW_AT_alignment 8. Its
// alignment (the max of its payload alignments) is not recoverable from the
// byte_size, which is why the compiler has to record it: stride is
// alignUp(34, 8) = 40, not alignUp(34, 34) = 66.
enum WideEnum {
  case wide(Wide)
  case pair(Int64, Int64)
  case none
}

// Two enums back to back: the offset of `second`, and the whole struct's size,
// are a direct readout of the enum's stride. Correct: 40 + 34 = 74.
struct EnumPair {
  var first: WideEnum
  var second: WideEnum
}

// A one-byte prefix followed by an enum: the offset of `payload` is a readout
// of the enum's alignment. Correct: the payload lands at alignUp(1, 8) = 8, so
// the struct's size is 8 + 34 = 42. A fabricated alignment of 34 rounded 1 up
// to 2 instead (the mask-based round-up is only valid for powers of two).
struct PrefixedEnum {
  var tag: Int8
  var payload: WideEnum
}

@inline(never)
func blackHole(_ x: Int64) {}

// An empty type is zero-sized but still 1-byte aligned.
struct Empty {}

enum ZeroSized {
  case only(Empty)
}

// One byte followed by a zero-sized enum: size 1, with the enum at offset 1.
struct PrefixedZeroSized {
  var tag: Int8
  var payload: ZeroSized
}

func f() {
  let w = WideEnum.wide(Wide())
  let pairs = EnumPair(first: .pair(7, 8), second: .pair(9, 10))
  let prefixed = PrefixedEnum(tag: 3, payload: .pair(11, 12))
  let zero = ZeroSized.only(Empty())
  let prefixedZero = PrefixedZeroSized(tag: 4, payload: .only(Empty()))

  if case .pair(let x, _) = pairs.second { blackHole(x) }
  if case .pair(let y, _) = prefixed.payload { blackHole(y) }
  if case .wide(let z) = w { blackHole(z.a) }
  if case .only = zero { blackHole(Int64(prefixedZero.tag)) }

  let s = StaticString("break here")
  print(s) // break here
}

f()
