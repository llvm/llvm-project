import gdb
import os
import re
import traceback
import sys

""" This module evaluates function parameters of those OMPD callbacks that need GDB API calls.
"""

""" Have the debugger print a string.
"""


def _print(*args):
    # args is a tuple with just one string element
    print_string = args[0]
    gdb.execute('printf "%s\n"' % args[0])


""" Look up the address of a global symbol in the target.
"""


def _sym_addr(*args):
    # args is a tuple consisting of thread_id and symbol_name
    thread_id = args[0]
    symbol_name = args[1]
    if thread_id >= 0:
        gdb.execute("thread %d\n" % thread_id, to_string=True)
    return int(gdb.parse_and_eval("&" + symbol_name))


""" Read string from the target and copy it into the provided buffer.
"""


def _read_string(*args):
    # args is a tuple with just the source address
    addr = args[0]
    try:
        buf = gdb.parse_and_eval("(unsigned char*)%li" % addr).string()
    except:
        traceback.print_exc()
    return buf


""" Read memory from the target and copy it into the provided buffer.
"""


def _read(*args):
    # args is a tuple consisting of address and number of bytes to be read
    addr = args[0]
    nbytes = args[1]
    # 	print("_read(%i,%i)"%(addr, nbytes))
    ret_buf = bytearray()
    # 	try:
    buf = gdb.parse_and_eval("(unsigned char*)%li" % addr)
    for i in range(nbytes):
        ret_buf.append(int(buf[i]))
    # 	except:
    # 		traceback.print_exc()
    return ret_buf


""" Get thread-specific context.
Return -1 if no match is found.
"""


def _thread_context(*args):
    # args is a tuple consisting of thread_id and the thread kind
    thread_id = args[1]
    pthread = False
    lwp = False
    if args[0] == 0:
        pthread = True
    else:
        lwp = True
    info = gdb.execute("info threads", to_string=True).splitlines()

    for line in info:
        if pthread:
            m = re.search(r"(0x[a-fA-F0-9]+)", line)
        elif lwp:
            m = re.search(r"\([^)]*?(\d+)[^)]*?\)", line)
        if m is None:
            continue
        pid = int(m.group(1), 0)
        if pid == thread_id:
            return int(line[2:6], 0)
    return -1


""" Test info threads / list threads / how to split output to get thread id 
and its size.
"""


def _test_threads(*args):
    info = gdb.execute("info threads", to_string=True).splitlines()
    for line in info[1:]:
        content = line.split()
        thread_id = None
        # fetch pointer to id
        if content[0].startswith("*"):
            thread_id = content[3]
        else:
            thread_id = content[2]
        sizeof_tid = sys.getsizeof(thread_id)
        print(sizeof_tid)
    print(info)
