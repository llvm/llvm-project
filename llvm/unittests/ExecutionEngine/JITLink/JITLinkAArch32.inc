/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Skeleton data structures                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

using namespace llvm::jitlink::aarch32;

namespace llvm {
namespace jitlink {
namespace aarch32 {
template <> struct FixupInfo<Arm_MovwAbsNC> : public FixupInfoArm {
    static constexpr uint32_t Opcode = 0x3000000;
    static constexpr uint32_t OpcodeMask = 0xff00000;
    static constexpr uint32_t ImmMask = 0xf0fff;
    static constexpr uint32_t RegMask = 0xf000;
};

template <> struct FixupInfo<Arm_MovtAbs> : public FixupInfoArm {
    static constexpr uint32_t Opcode = 0x3400000;
    static constexpr uint32_t OpcodeMask = 0xff00000;
    static constexpr uint32_t ImmMask = 0xf0fff;
    static constexpr uint32_t RegMask = 0xf000;
};

template <> struct FixupInfo<Thumb_MovwAbsNC> : public FixupInfoThumb {
    static constexpr HalfWords Opcode {0xf240, 0x0};
    static constexpr HalfWords OpcodeMask {0xfbf0, 0x8000};
    static constexpr HalfWords ImmMask {0x40f, 0x70ff};
    static constexpr HalfWords RegMask {0x0, 0xf00};
};

template <> struct FixupInfo<Thumb_MovtAbs> : public FixupInfoThumb {
    static constexpr HalfWords Opcode {0xf2c0, 0x0};
    static constexpr HalfWords OpcodeMask {0xfbf0, 0x8000};
    static constexpr HalfWords ImmMask {0x40f, 0x70ff};
    static constexpr HalfWords RegMask {0x0, 0xf00};
};

template <> struct FixupInfo<Thumb_MovwPrelNC> : public FixupInfoThumb {
    static constexpr HalfWords Opcode {0xf240, 0x0};
    static constexpr HalfWords OpcodeMask {0xfbf0, 0x8000};
    static constexpr HalfWords ImmMask {0x40f, 0x70ff};
    static constexpr HalfWords RegMask {0x0, 0xf00};
};

template <> struct FixupInfo<Thumb_MovtPrel> : public FixupInfoThumb {
    static constexpr HalfWords Opcode {0xf2c0, 0x0};
    static constexpr HalfWords OpcodeMask {0xfbf0, 0x8000};
    static constexpr HalfWords ImmMask {0x40f, 0x70ff};
    static constexpr HalfWords RegMask {0x0, 0xf00};
};

} //aarch32
} //jitlink
} //llvm
