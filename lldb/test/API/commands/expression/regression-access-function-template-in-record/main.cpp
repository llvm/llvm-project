namespace n {
template <class> class a {};
template <class b> struct shared_ptr {
  template <class...>
  static void make_shared() { //%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
    typedef a<b> c;
    c d;
  }
};
} // namespace n
int main() { n::shared_ptr<int>::make_shared(); }
