
// library_c_api.h - Mock type erased C API for library
#include <cstddef>

typedef void *Handle;

typedef Handle FooHandle;
typedef Handle BarHandle;

// To minimize API surface, the C API has a single type for arrays of handles
struct HandleArray {
  size_t size;
  Handle *data;
};

struct SessionInfo {
  // .. as a consequence we end up with cases where type information is only
  // encoded in either variable names, or documentation
  HandleArray foos;
  HandleArray bars;
};

extern "C" {
void InitSession();
void StopSession();

void ReadSessionInfo(SessionInfo *into);
void FreeSessionInfo(SessionInfo *info);
}

// library.cpp
#include <memory>
#include <vector>

#include <stdlib.h>

struct Foo {
  int foo;
};
struct Bar {
  bool bar;
};

static std::vector<std::unique_ptr<Foo>> g_foos;
static std::vector<std::unique_ptr<Bar>> g_bars;

extern "C" {
void InitSession() {
  g_foos.emplace_back(new Foo{.foo = 10});
  g_foos.emplace_back(new Foo{.foo = 20});
  g_bars.emplace_back(new Bar{.bar = false});
}
void StopSession() {
  g_foos.clear();
  g_bars.clear();
}

void ReadSessionInfo(SessionInfo *into) {
  into->foos.size = g_foos.size();
  into->foos.data = (Handle *)malloc(sizeof(Handle) * g_foos.size());
  for (size_t i = 0; i < g_foos.size(); i++) {
    into->foos.data[i] = g_foos[i].get();
  }

  into->bars.size = g_bars.size();
  into->bars.data = (Handle *)malloc(sizeof(Handle) * g_bars.size());
  for (size_t i = 0; i < g_bars.size(); i++) {
    into->bars.data[i] = g_bars[i].get();
  }
}
void FreeSessionInfo(SessionInfo *info) {
  free(info->foos.data);
  free(info->bars.data);

  info->foos.size = 0;
  info->foos.data = NULL;
  info->bars.size = 0;
  info->bars.data = NULL;
}
}

// app.cpp - being debugged by library.cpp authors

int main() {
  InitSession();

  SessionInfo info;
  ReadSessionInfo(&info);
  FreeSessionInfo(&info); // break here

  StopSession();
  return 0;
}
