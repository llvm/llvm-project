namespace {
template <typename T1, typename T2> struct Temp { int x; };
// This emits the 'Temp' template from this TU.
Temp<int, float> Template2;
} // namespace

int other() { return Template2.x; }
