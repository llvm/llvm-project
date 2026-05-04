#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build="${BUILD_DIR:-$repo/build-libcxx-llvm}"
base_c_compiler="${CC:-/usr/bin/clang}"
base_compiler="${CXX:-/usr/bin/clang++}"
lit_jobs="${LIT_JOBS:-4}"
host_arch="$(uname -m)"
multiarch_triple=""
configure_c_flags="${CFLAGS:-}"
configure_cxx_flags="${CXXFLAGS:-}"
compiler_is_clang=0

case "$host_arch" in
  x86_64)
    compiler_rt_arch=x86_64
    ;;
  aarch64|arm64)
    compiler_rt_arch=aarch64
    ;;
  *)
    compiler_rt_arch="$host_arch"
    ;;
esac

if "$base_compiler" -dM -E -x c++ /dev/null 2>/dev/null | grep -q '__clang__'; then
  compiler_is_clang=1
fi

cd "$repo"

for candidate in \
  "$(
    case "$host_arch" in
      x86_64) echo x86_64-linux-gnu ;;
      aarch64|arm64) echo aarch64-linux-gnu ;;
      riscv64) echo riscv64-linux-gnu ;;
      *) ;;
    esac
  )" \
  "$("$base_c_compiler" -print-multiarch 2>/dev/null || true)" \
  "$("$base_compiler" -print-multiarch 2>/dev/null || true)" \
  "$("$base_c_compiler" -dumpmachine 2>/dev/null || true)" \
  "$("$base_compiler" -dumpmachine 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -d "/usr/include/$candidate" ]]; then
    multiarch_triple="$candidate"
    break
  fi
done

if [[ -z "$multiarch_triple" ]]; then
  multiarch_triple="$(find /usr/include -mindepth 1 -maxdepth 1 -type d -name '*-linux-gnu' -exec test -d '{}/asm' ';' -print -quit | xargs -r basename)"
fi

if [[ -n "$multiarch_triple" ]]; then
  configure_c_flags="${configure_c_flags:+$configure_c_flags }-idirafter/usr/include/$multiarch_triple"
  configure_cxx_flags="${configure_cxx_flags:+$configure_cxx_flags }-idirafter/usr/include/$multiarch_triple"
fi

if [[ ! -f "$build/CMakeCache.txt" ]]; then
  mkdir -p "$build"
  cmake \
    -S "$repo/runtimes" \
    -B "$build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$base_c_compiler" \
    -DCMAKE_CXX_COMPILER="$base_compiler" \
    -DCMAKE_C_FLAGS="$configure_c_flags" \
    -DCMAKE_CXX_FLAGS="$configure_cxx_flags" \
    -DCMAKE_C_COMPILER_LAUNCHER= \
    -DCMAKE_CXX_COMPILER_LAUNCHER= \
    -DLIBUNWIND_ENABLE_SHARED=OFF \
    -DLIBUNWIND_ENABLE_STATIC=ON \
    -DLIBCXXABI_ENABLE_SHARED=OFF \
    -DLIBCXXABI_ENABLE_STATIC=ON \
    -DLIBCXXABI_ENABLE_STATIC_UNWINDER=ON \
    -DLIBCXX_ENABLE_SHARED=OFF \
    -DLIBCXX_ENABLE_STATIC=ON \
    -DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON \
    -DLIBCXX_ENABLE_FILESYSTEM=OFF \
    -DLIBCXX_ENABLE_WIDE_CHARACTERS=OFF \
    -DLLVM_LIBC_INCLUDE_SCUDO=ON \
    -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF \
    -DRUNTIMES_USE_LIBC=llvm-libc \
    -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt;libunwind;libcxxabi;libcxx" \
    -DLIBCXX_INCLUDE_TESTS=ON \
    -DLIBCXX_INCLUDE_BENCHMARKS=OFF
fi

builtins_target="clang_rt.builtins-$compiler_rt_arch"
builtins_target_candidates="$(ninja -C "$build" -t targets all | sed -n 's/:.*//; /^clang_rt\.builtins-/p; /^libclang_rt\.builtins-.*\.a$/p')"
if ! grep -qx "$builtins_target" <<<"$builtins_target_candidates"; then
  alt_target="libclang_rt.builtins-$compiler_rt_arch.a"
  if grep -qx "$alt_target" <<<"$builtins_target_candidates"; then
    builtins_target="$alt_target"
  else
    builtins_target="$(head -n 1 <<<"$builtins_target_candidates")"
    if [[ -z "$builtins_target" ]]; then
      echo "unable to locate a compiler-rt builtins target in $build" >&2
      exit 1
    fi
  fi
fi

root="$(mktemp -d "$build/run-sandbox.XXXXXX")"
launcher_shims="$root/shims"
sysroot="$root/sysroot"
wrapper="$root/clang++-llvm-libc"
lit_wrapper="$root/llvm-lit-fork"
empty_include="$root/empty-usr-local-include"
fortify_shim_c="$root/fortify-shim.c"
fortify_shim_o="$root/fortify-shim.o"
multiarch_include_dir=""
asm_source_dir=""

cleanup() {
  rm -rf "$root"
}
trap cleanup EXIT

mkdir -p "$launcher_shims"
cat > "$launcher_shims/sccache" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
chmod +x "$launcher_shims/sccache"
export PATH="$launcher_shims:$PATH"

if (($# == 0)); then
  targets=("$build/libcxx/test")
else
  targets=("$@")
fi

ninja -C "$build" \
  libc/include/generate-libc-headers \
  crt1.o \
  libc.a \
  libm.a \
  lib/libunwind.a \
  "$builtins_target" \
  cxx-test-depends

builtins="$build/compiler-rt/lib/linux/libclang_rt.builtins-$compiler_rt_arch.a"
if [[ ! -f "$builtins" ]]; then
  builtins="$(find "$build/compiler-rt/lib" -name "libclang_rt.builtins-$compiler_rt_arch.a" -print -quit)"
fi
if [[ -z "${builtins:-}" || ! -f "$builtins" ]]; then
  builtins="$(find "$build/compiler-rt/lib" -name 'libclang_rt.builtins-*.a' -print -quit)"
  if [[ -z "${builtins:-}" ]]; then
    echo "unable to locate compiler-rt builtins archive under $build/compiler-rt/lib" >&2
    exit 1
  fi
fi

mkdir -p "$sysroot/usr/include" "$empty_include"

# Preserve Linux UAPI headers, but do not copy libc++ into /usr/include.
if [[ -d /usr/include/linux ]]; then
  cp -a /usr/include/linux "$sysroot/usr/include/"
fi
if [[ -d /usr/include/asm-generic ]]; then
  cp -a /usr/include/asm-generic "$sysroot/usr/include/"
fi
if [[ -n "$multiarch_triple" ]]; then
  multiarch_include_dir="$sysroot/usr/include/$multiarch_triple"
  mkdir -p "$multiarch_include_dir"
fi
if [[ -n "$multiarch_triple" && -d "/usr/include/$multiarch_triple/asm" ]]; then
  asm_source_dir="$(readlink -f "/usr/include/$multiarch_triple/asm")"
elif [[ -d /usr/include/asm ]]; then
  asm_source_dir="$(readlink -f /usr/include/asm)"
fi
if [[ -n "$asm_source_dir" ]]; then
  mkdir -p "${multiarch_include_dir:-$sysroot/usr/include}/asm"
  cp -a "$asm_source_dir/." "${multiarch_include_dir:-$sysroot/usr/include}/asm/"
fi
cp -a "$build/libc/include/." "$sysroot/usr/include/"

crtbegin="$("$base_compiler" --print-file-name=crtbeginT.o)"
crtend="$("$base_compiler" --print-file-name=crtend.o)"

cat > "$fortify_shim_c" <<'EOF'
typedef __SIZE_TYPE__ size_t;

__attribute__((noreturn)) static void __llvm_libcxx_test_chk_fail(void) {
  __builtin_trap();
}

static void *__llvm_libcxx_test_memcpy(void *dst, const void *src, size_t len) {
  unsigned char *out = (unsigned char *)dst;
  const unsigned char *in = (const unsigned char *)src;
  for (size_t i = 0; i < len; ++i)
    out[i] = in[i];
  return dst;
}

static void *__llvm_libcxx_test_memmove(void *dst, const void *src, size_t len) {
  unsigned char *out = (unsigned char *)dst;
  const unsigned char *in = (const unsigned char *)src;
  if (out == in || len == 0)
    return dst;
  if (out < in || out >= in + len) {
    for (size_t i = 0; i < len; ++i)
      out[i] = in[i];
  } else {
    for (size_t i = len; i != 0; --i)
      out[i - 1] = in[i - 1];
  }
  return dst;
}

static void *__llvm_libcxx_test_memset(void *dst, int value, size_t len) {
  unsigned char *out = (unsigned char *)dst;
  unsigned char byte = (unsigned char)value;
  for (size_t i = 0; i < len; ++i)
    out[i] = byte;
  return dst;
}

void *__memcpy_chk(void *dst, const void *src, size_t len, size_t dstlen) {
  if (len > dstlen)
    __llvm_libcxx_test_chk_fail();
  return __llvm_libcxx_test_memcpy(dst, src, len);
}

void *__memmove_chk(void *dst, const void *src, size_t len, size_t dstlen) {
  if (len > dstlen)
    __llvm_libcxx_test_chk_fail();
  return __llvm_libcxx_test_memmove(dst, src, len);
}

void *__memset_chk(void *dst, int value, size_t len, size_t dstlen) {
  if (len > dstlen)
    __llvm_libcxx_test_chk_fail();
  return __llvm_libcxx_test_memset(dst, value, len);
}
EOF

"$base_compiler" \
  -c \
  -x c \
  -fno-stack-protector \
  -ffreestanding \
  "$fortify_shim_c" \
  -o "$fortify_shim_o"

cat > "$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail

base_compiler=$base_compiler
sysroot=$sysroot
repo=$repo
build=$build
crtbegin=$crtbegin
crtend=$crtend
builtins=$builtins
fortify_shim_o=$fortify_shim_o
multiarch_include_dir=$multiarch_include_dir
compiler_is_clang=$compiler_is_clang

linking=1
for arg in "\$@"; do
  case "\$arg" in
    -c|-E|-S|-fsyntax-only|-emit-ast)
      linking=0
      ;;
  esac
done

common_args=(
  -nostdlibinc
  -isystem "\$sysroot/usr/include"
  -fno-stack-protector
  -Wno-missing-braces
)
if (( compiler_is_clang )); then
  # The sandboxed sysroot setup can trigger Clang's GCC-install-dir warning,
  # and libc++'s config probes use -Werror while checking -std= flags.
  common_args+=(-Wno-unknown-warning-option -Wno-gcc-install-dir-libstdcxx)
fi
if [[ -n "\$multiarch_include_dir" ]]; then
  common_args+=(-isystem "\$multiarch_include_dir")
fi

if (( linking )); then
  exec "\$base_compiler" \
    "\$@" \
    "\${common_args[@]}" \
    -nodefaultlibs \
    -nostartfiles \
    -static \
    "$build/libc/startup/linux/crt1.o" \
    "\$crtbegin" \
    -L "$build/libc/lib" \
    -lunwind \
    -lc \
    -lm \
    "\$fortify_shim_o" \
    "\$builtins" \
    "\$crtend"
else
  exec "\$base_compiler" "\$@" "\${common_args[@]}"
fi
EOF

chmod +x "$wrapper"

cat > "$lit_wrapper" <<EOF
#!/usr/bin/env python3
import multiprocessing as mp
import runpy
import sys

mp.set_start_method("fork")
sys.argv[0] = "$build/bin/llvm-lit"
runpy.run_path("$build/bin/llvm-lit", run_name="__main__")
EOF

chmod +x "$lit_wrapper"

bwrap \
  --ro-bind / / \
  --bind "$repo" "$repo" \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --chdir "$repo" \
  --ro-bind "$sysroot/usr/include" /usr/include \
  --ro-bind "$empty_include" /usr/local/include \
  "$lit_wrapper" \
  -sv \
  "-j$lit_jobs" \
  --param "compiler=$wrapper" \
  "${targets[@]}"
