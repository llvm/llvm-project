#![feature(repr128)]
#![feature(rustc_attrs)]
#![feature(const_option)]

use core::num::{NonZeroI128, NonZeroU32};

/// This file was manually compiled with rustc as object file
/// obj2yaml tool was used to convert this to main.yaml
/// This is done in order to make the test portable since LLVM codebase tests don't have setup to compile Rust programs
/// no_std , no_main is used in order to make the object file as small as possible eliminating extra symbols from standard library
/// static global variables are used because they can be inspected on object file without starting the process

/// Command:
/// rustc -g --emit=obj --crate-type=bin -C panic=abort -C link-arg=-nostdlib main.rs && obj2yaml main.o -o main.yaml
use core::ptr::NonNull;

pub enum CLikeEnumDefault {
    A = 2,
    B = 10,
}

#[repr(u8)]
pub enum CLikeEnumReprU8 {
    VariantA,
    VariantB,
    VariantC,
}

#[repr(u32)]
pub enum CLikeEnumReprU32 {
    VariantA = 1,
    VariantB = 2,
    VariantC = 3,
}

pub enum EnumWithTuples {
    A(u8),
    B(u16),
    C(u32),
    D(usize),
    AA(u8, u8),
    BB(u16, u16),
    BC(u16, u32),
    CC(u32, u32),
    // no DD on purpose to have D = CC in size
}

pub enum EnumWithStructs {
    A(Struct1),
    B(Struct2),
}

#[repr(usize)]
pub enum MixedEnum {
    A,
    B(i32),
    C(u8, i32),
    D(Option<Struct2>),
    E(EnumWithStructs),
}

pub struct Struct1 {
    field: i32,
}

pub struct Struct2 {
    field: u32,
    inner: Struct1,
}

pub struct NonNullHolder {
    inner: Option<NonNull<u64>>,
}

pub enum NicheLayoutEnum {
    Tag1,
    Data { my_data: CLikeEnumDefault },
    Tag2,
}

// The following three types will use a niche layout once
pub enum NicheLayoutWithFields1<'a> {
    A(&'a u8, u32),
    B(u32),
}

pub enum NicheLayoutWithFields2 {
    A(NonZeroU32, u64),
    B(u64),
}

pub enum NicheLayoutWithFields3 {
    A(u8, bool),
    B(u8),
    C(bool),
    D(u8),
    E(u8),
    F,
}

#[repr(i128)]
enum DirectTag128 {
    A(u32),
    B(u32),
}

static CLIKE_DEFAULT_A: CLikeEnumDefault = CLikeEnumDefault::A;
static CLIKE_DEFAULT_B: CLikeEnumDefault = CLikeEnumDefault::B;

static CLIKE_U8_A: CLikeEnumReprU8 = CLikeEnumReprU8::VariantA;
static CLIKE_U8_B: CLikeEnumReprU8 = CLikeEnumReprU8::VariantB;
static CLIKE_U8_C: CLikeEnumReprU8 = CLikeEnumReprU8::VariantC;

static CLIKE_U32_A: CLikeEnumReprU32 = CLikeEnumReprU32::VariantA;
static CLIKE_U32_B: CLikeEnumReprU32 = CLikeEnumReprU32::VariantB;
static CLIKE_U32_C: CLikeEnumReprU32 = CLikeEnumReprU32::VariantC;

static ENUM_WITH_TUPLES_A: EnumWithTuples = EnumWithTuples::A(13);
static ENUM_WITH_TUPLES_AA: EnumWithTuples = EnumWithTuples::AA(13, 37);
static ENUM_WITH_TUPLES_B: EnumWithTuples = EnumWithTuples::B(37);
static ENUM_WITH_TUPLES_BB: EnumWithTuples = EnumWithTuples::BB(37, 5535);
static ENUM_WITH_TUPLES_BC: EnumWithTuples = EnumWithTuples::BC(65000, 165000);
static ENUM_WITH_TUPLES_C: EnumWithTuples = EnumWithTuples::C(31337);
static ENUM_WITH_TUPLES_CC: EnumWithTuples = EnumWithTuples::CC(31337, 87236);
static ENUM_WITH_TUPLES_D: EnumWithTuples = EnumWithTuples::D(123456789012345678);

static MIXED_ENUM_A: MixedEnum = MixedEnum::A;
static MIXED_ENUM_B: MixedEnum = MixedEnum::B(-10);
static MIXED_ENUM_C: MixedEnum = MixedEnum::C(254, -254);
static MIXED_ENUM_D_NONE: MixedEnum = MixedEnum::D(None);
static MIXED_ENUM_D_SOME: MixedEnum = MixedEnum::D(Some(Struct2 {
    field: 123456,
    inner: Struct1 { field: 123 },
}));

static NICHE_W_FIELDS_1_A: NicheLayoutWithFields1 = NicheLayoutWithFields1::A(&77, 7);
static NICHE_W_FIELDS_1_B: NicheLayoutWithFields1 = NicheLayoutWithFields1::B(99);
static NICHE_W_FIELDS_2_A: NicheLayoutWithFields2 =
    NicheLayoutWithFields2::A(NonZeroU32::new(800).unwrap(), 900);
static NICHE_W_FIELDS_2_B: NicheLayoutWithFields2 = NicheLayoutWithFields2::B(1000);
static NICHE_W_FIELDS_3_A: NicheLayoutWithFields3 = NicheLayoutWithFields3::A(137, true);
static NICHE_W_FIELDS_3_B: NicheLayoutWithFields3 = NicheLayoutWithFields3::B(12);
static NICHE_W_FIELDS_3_C: NicheLayoutWithFields3 = NicheLayoutWithFields3::C(false);
static NICHE_W_FIELDS_3_D: NicheLayoutWithFields3 = NicheLayoutWithFields3::D(34);
static NICHE_W_FIELDS_3_E: NicheLayoutWithFields3 = NicheLayoutWithFields3::E(56);
static NICHE_W_FIELDS_3_F: NicheLayoutWithFields3 = NicheLayoutWithFields3::F;

static DIRECT_TAG_128_A: DirectTag128 = DirectTag128::A(12345);
static DIRECT_TAG_128_B: DirectTag128 = DirectTag128::B(6789);

pub fn main() {
    let niche_w_fields_1_a: NicheLayoutWithFields1 = NicheLayoutWithFields1::A(&77, 7);
    let niche_w_fields_1_b: NicheLayoutWithFields1 = NicheLayoutWithFields1::B(99);
    let direct_tag_128_a: DirectTag128 = DirectTag128::A(0xF1F2);
    let direct_tag_128_b: DirectTag128 = DirectTag128::B(0xF3F4);
    let non_null = unsafe {
        NonNullHolder {
            inner: NonNull::new(12345 as *mut u64),
        }
    };
}
