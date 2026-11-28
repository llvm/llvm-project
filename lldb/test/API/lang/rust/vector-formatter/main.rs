/// This file was manually compiled with rustc and used to generate a core file.
///
/// This is done in order to make the test portable since LLVM codebase tests
/// don't have setup to compile Rust programs.
///
/// Compilation Command:
/// rustc -g --crate-type=bin -C panic=abort main.rs
///
/// To generate core file:
/// lldb main -o "br set -f main.rs -l 15" -o "run" -o "process save-core -s modified-memory main.core" -o "quit"

fn main() {
    let vector = vec![1, 2, 3];

    if vector[0] == 17 { // Set breakpoint here (line 15)
        println!("condition was true");
    } else {
        println!("condition was false");
    }
}
