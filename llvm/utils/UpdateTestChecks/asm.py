from __future__ import print_function
import re
import sys

from . import common

if sys.version_info[0] > 2:
  class string:
    expandtabs = str.expandtabs
else:
  import string

# RegEx: this is where the magic happens.

##### Assembly parser

ASM_FUNCTION_X86_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n(?:\s*\.?Lfunc_begin[^:\n]*:\n)?[^:]*?'
    r'(?P<body>^##?[ \t]+[^:]+:.*?)\s*'
    r'^\s*(?:[^:\n]+?:\s*\n\s*\.size|\.cfi_endproc|\.globl|\.comm|\.(?:sub)?section|#+ -- End function)',
    flags=(re.M | re.S))

ASM_FUNCTION_ARM_RE = re.compile(
        r'^(?P<func>[0-9a-zA-Z_]+):\n' # f: (name of function)
        r'\s+\.fnstart\n' # .fnstart
        r'(?P<body>.*?)\n' # (body of the function)
        r'.Lfunc_end[0-9]+:', # .Lfunc_end0: or # -- End function
        flags=(re.M | re.S))

ASM_FUNCTION_AARCH64_RE = re.compile(
     r'^_?(?P<func>[^:]+):[ \t]*\/\/[ \t]*@(?P=func)\n'
     r'(?:[ \t]+.cfi_startproc\n)?'  # drop optional cfi noise
     r'(?P<body>.*?)\n'
     # This list is incomplete
     r'.Lfunc_end[0-9]+:\n',
     flags=(re.M | re.S))

ASM_FUNCTION_AMDGPU_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*;+[ \t]*@(?P=func)\n[^:]*?'
    r'(?P<body>.*?)\n' # (body of the function)
    # This list is incomplete
    r'^\s*(\.Lfunc_end[0-9]+:\n|\.section)',
    flags=(re.M | re.S))

ASM_FUNCTION_HEXAGON_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*//[ \t]*@(?P=func)\n[^:]*?'
    r'(?P<body>.*?)\n' # (body of the function)
    # This list is incomplete
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_MIPS_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n[^:]*?' # f: (name of func)
    r'(?:^[ \t]+\.(frame|f?mask|set).*?\n)+'  # Mips+LLVM standard asm prologue
    r'(?P<body>.*?)\n'                        # (body of the function)
    # Mips+LLVM standard asm epilogue
    r'(?:(^[ \t]+\.set[^\n]*?\n)*^[ \t]+\.end.*?\n)'
    r'(\$|\.L)func_end[0-9]+:\n',             # $func_end0: (mips32 - O32) or
                                              # .Lfunc_end0: (mips64 - NewABI)
    flags=(re.M | re.S))

ASM_FUNCTION_MSP430_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*;+[ \t]*@(?P=func)\n[^:]*?'
    r'(?P<body>.*?)\n'
    r'(\$|\.L)func_end[0-9]+:\n',             # $func_end0:
    flags=(re.M | re.S))

ASM_FUNCTION_PPC_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n'
    r'.*?'
    r'\.Lfunc_begin[0-9]+:\n'
    r'(?:[ \t]+.cfi_startproc\n)?'
    r'(?:\.Lfunc_[gl]ep[0-9]+:\n(?:[ \t]+.*?\n)*)*'
    r'(?P<body>.*?)\n'
    # This list is incomplete
    r'(?:^[ \t]*(?:\.long[ \t]+[^\n]+|\.quad[ \t]+[^\n]+)\n)*'
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_RISCV_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n(?:\s*\.?Lfunc_begin[^:\n]*:\n)?[^:]*?'
    r'(?P<body>^##?[ \t]+[^:]+:.*?)\s*'
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_LANAI_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*!+[ \t]*@(?P=func)\n'
    r'(?:[ \t]+.cfi_startproc\n)?'  # drop optional cfi noise
    r'(?P<body>.*?)\s*'
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_SPARC_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*!+[ \t]*@(?P=func)\n'
    r'(?P<body>.*?)\s*'
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_SYSTEMZ_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n'
    r'[ \t]+.cfi_startproc\n'
    r'(?P<body>.*?)\n'
    r'.Lfunc_end[0-9]+:\n',
    flags=(re.M | re.S))

ASM_FUNCTION_AARCH64_DARWIN_RE = re.compile(
     r'^_(?P<func>[^:]+):[ \t]*;[ \t]@(?P=func)\n'
     r'([ \t]*.cfi_startproc\n[\s]*)?'
     r'(?P<body>.*?)'
     r'([ \t]*.cfi_endproc\n[\s]*)?'
     r'^[ \t]*;[ \t]--[ \t]End[ \t]function',
     flags=(re.M | re.S))

ASM_FUNCTION_ARM_DARWIN_RE = re.compile(
     r'^[ \t]*\.globl[ \t]*_(?P<func>[^ \t])[ \t]*@[ \t]--[ \t]Begin[ \t]function[ \t](?P=func)'
     r'(?P<directives>.*?)'
     r'^_(?P=func):\n[ \t]*'
     r'(?P<body>.*?)'
     r'^[ \t]*@[ \t]--[ \t]End[ \t]function',
     flags=(re.M | re.S ))

ASM_FUNCTION_ARM_MACHO_RE = re.compile(
     r'^_(?P<func>[^:]+):[ \t]*\n'
     r'([ \t]*.cfi_startproc\n[ \t]*)?'
     r'(?P<body>.*?)\n'
     r'[ \t]*\.cfi_endproc\n',
     flags=(re.M | re.S))

ASM_FUNCTION_ARM_IOS_RE = re.compile(
     r'^_(?P<func>[^:]+):[ \t]*\n'
     r'^Lfunc_begin(?P<id>[0-9][1-9]*):\n'
     r'(?P<body>.*?)'
     r'^Lfunc_end(?P=id):\n'
     r'^[ \t]*@[ \t]--[ \t]End[ \t]function',
     flags=(re.M | re.S))

ASM_FUNCTION_WASM32_RE = re.compile(
    r'^_?(?P<func>[^:]+):[ \t]*#+[ \t]*@(?P=func)\n'
    r'(?P<body>.*?)\n'
    r'^\s*(\.Lfunc_end[0-9]+:\n|end_function)',
    flags=(re.M | re.S))


SCRUB_LOOP_COMMENT_RE = re.compile(
    r'# =>This Inner Loop Header:.*|# in Loop:.*', flags=re.M)

SCRUB_X86_SHUFFLES_RE = (
    re.compile(
        r'^(\s*\w+) [^#\n]+#+ ((?:[xyz]mm\d+|mem)( \{%k\d+\}( \{z\})?)? = .*)$',
        flags=re.M))
SCRUB_X86_SPILL_RELOAD_RE = (
    re.compile(
        r'-?\d+\(%([er])[sb]p\)(.*(?:Spill|Reload))$',
        flags=re.M))
SCRUB_X86_SP_RE = re.compile(r'\d+\(%(esp|rsp)\)')
SCRUB_X86_RIP_RE = re.compile(r'[.\w]+\(%rip\)')
SCRUB_X86_LCP_RE = re.compile(r'\.LCPI[0-9]+_[0-9]+')
SCRUB_X86_RET_RE = re.compile(r'ret[l|q]')

def scrub_asm_x86(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Detect shuffle asm comments and hide the operands in favor of the comments.
  asm = SCRUB_X86_SHUFFLES_RE.sub(r'\1 {{.*#+}} \2', asm)
  # Detect stack spills and reloads and hide their exact offset and whether
  # they used the stack pointer or frame pointer.
  asm = SCRUB_X86_SPILL_RELOAD_RE.sub(r'{{[-0-9]+}}(%\1{{[sb]}}p)\2', asm)
  # Generically match the stack offset of a memory operand.
  asm = SCRUB_X86_SP_RE.sub(r'{{[0-9]+}}(%\1)', asm)
  if getattr(args, 'x86_scrub_rip', False):
    # Generically match a RIP-relative memory operand.
    asm = SCRUB_X86_RIP_RE.sub(r'{{.*}}(%rip)', asm)
  # Generically match a LCP symbol.
  asm = SCRUB_X86_LCP_RE.sub(r'{{\.LCPI.*}}', asm)
  if getattr(args, 'extra_scrub', False):
    # Avoid generating different checks for 32- and 64-bit because of 'retl' vs 'retq'.
    asm = SCRUB_X86_RET_RE.sub(r'ret{{[l|q]}}', asm)
  # Strip kill operands inserted into the asm.
  asm = common.SCRUB_KILL_COMMENT_RE.sub('', asm)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_amdgpu(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_arm_eabi(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip kill operands inserted into the asm.
  asm = common.SCRUB_KILL_COMMENT_RE.sub('', asm)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_hexagon(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_powerpc(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Stripe unimportant comments, but leave the token '#' in place.
  asm = SCRUB_LOOP_COMMENT_RE.sub(r'#', asm)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_mips(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_msp430(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_riscv(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_lanai(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_sparc(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_systemz(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def scrub_asm_wasm32(asm, args):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  asm = common.SCRUB_WHITESPACE_RE.sub(r' ', asm)
  # Expand the tabs used for indentation.
  asm = string.expandtabs(asm, 2)
  # Strip trailing whitespace.
  asm = common.SCRUB_TRAILING_WHITESPACE_RE.sub(r'', asm)
  return asm

def get_triple_from_march(march):
  triples = {
      'amdgcn': 'amdgcn',
      'r600': 'r600',
      'mips': 'mips',
      'sparc': 'sparc',
      'hexagon': 'hexagon',
  }
  for prefix, triple in triples.items():
    if march.startswith(prefix):
      return triple
  print("Cannot find a triple. Assume 'x86'", file=sys.stderr)
  return 'x86'

def build_function_body_dictionary_for_triple(args, raw_tool_output, triple, prefixes, func_dict):
  target_handlers = {
      'i686': (scrub_asm_x86, ASM_FUNCTION_X86_RE),
      'x86': (scrub_asm_x86, ASM_FUNCTION_X86_RE),
      'i386': (scrub_asm_x86, ASM_FUNCTION_X86_RE),
      'aarch64': (scrub_asm_arm_eabi, ASM_FUNCTION_AARCH64_RE),
      'aarch64-apple-darwin': (scrub_asm_arm_eabi, ASM_FUNCTION_AARCH64_DARWIN_RE),
      'hexagon': (scrub_asm_hexagon, ASM_FUNCTION_HEXAGON_RE),
      'r600': (scrub_asm_amdgpu, ASM_FUNCTION_AMDGPU_RE),
      'amdgcn': (scrub_asm_amdgpu, ASM_FUNCTION_AMDGPU_RE),
      'arm': (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_RE),
      'arm64': (scrub_asm_arm_eabi, ASM_FUNCTION_AARCH64_RE),
      'arm64-apple-ios': (scrub_asm_arm_eabi, ASM_FUNCTION_AARCH64_DARWIN_RE),
      'armv7-apple-ios' : (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_IOS_RE),
      'armv7-apple-darwin': (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_DARWIN_RE),
      'thumb': (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_RE),
      'thumb-macho': (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_MACHO_RE),
      'thumbv5-macho': (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_MACHO_RE),
      'thumbv7-apple-ios' : (scrub_asm_arm_eabi, ASM_FUNCTION_ARM_IOS_RE),
      'mips': (scrub_asm_mips, ASM_FUNCTION_MIPS_RE),
      'msp430': (scrub_asm_msp430, ASM_FUNCTION_MSP430_RE),
      'ppc32': (scrub_asm_powerpc, ASM_FUNCTION_PPC_RE),
      'powerpc': (scrub_asm_powerpc, ASM_FUNCTION_PPC_RE),
      'riscv32': (scrub_asm_riscv, ASM_FUNCTION_RISCV_RE),
      'riscv64': (scrub_asm_riscv, ASM_FUNCTION_RISCV_RE),
      'lanai': (scrub_asm_lanai, ASM_FUNCTION_LANAI_RE),
      'sparc': (scrub_asm_sparc, ASM_FUNCTION_SPARC_RE),
      's390x': (scrub_asm_systemz, ASM_FUNCTION_SYSTEMZ_RE),
      'wasm32': (scrub_asm_wasm32, ASM_FUNCTION_WASM32_RE),
  }
  handler = None
  best_prefix = ''
  for prefix, s in target_handlers.items():
    if triple.startswith(prefix) and len(prefix) > len(best_prefix):
      handler = s
      best_prefix = prefix

  if handler is None:
    raise KeyError('Triple %r is not supported' % (triple))

  scrubber, function_re = handler
  common.build_function_body_dictionary(
          function_re, scrubber, [args], raw_tool_output, prefixes,
          func_dict, args.verbose, False)

##### Generator of assembly CHECK lines

def add_asm_checks(output_lines, comment_marker, prefix_list, func_dict, func_name):
  # Label format is based on ASM string.
  check_label_format = '{} %s-LABEL: %s%s:'.format(comment_marker)
  common.add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, check_label_format, True, False)
