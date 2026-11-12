extern "C" int __attribute__((naked)) __AArch64ADRPThunk_step_here() {
    asm (
      "adrp x16, step_here\n"
      "add x16, x16, :lo12:step_here\n"
      "br x16"
    );
}

extern "C" __attribute__((used)) int step_here() {
    return 47;
}

int main() {
  return __AArch64ADRPThunk_step_here();
}
