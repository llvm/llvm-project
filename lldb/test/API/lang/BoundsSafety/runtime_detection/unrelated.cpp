namespace Unrelated {
int value() { return 42; }
} // namespace Unrelated

extern "C" int unrelated(void) { return Unrelated::value(); }
