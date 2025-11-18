/// Command:
/// rustc -g --emit=obj --crate-type=bin -C panic=abort -C link-arg=-nostdlib main.rs && obj2yaml main.o -o main.yaml

pub enum A {
    A(B),
}

pub enum B {
    B(u8),
}

static ENUM_INSTANCE: A = A::A(B::B(10));

pub fn main() {
}
