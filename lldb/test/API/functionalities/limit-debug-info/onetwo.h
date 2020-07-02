#ifdef IN_ONE
#define ONE_API LLDB_DYLIB_EXPORT
#else
#define ONE_API LLDB_DYLIB_IMPORT
#endif

#ifdef IN_TWO
#define TWO_API LLDB_DYLIB_EXPORT
#else
#define TWO_API LLDB_DYLIB_IMPORT
#endif

struct ONE_API One {
  int one = 142;
  constexpr One() = default;
  virtual ~One();
};

struct TWO_API Two : One {
  int two = 242;
  constexpr Two() = default;
  ~Two() override;
};
