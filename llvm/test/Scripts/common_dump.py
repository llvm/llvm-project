def dataToHex(d):
    """ Convert the raw data in 'd' to an hex string with a space every 4 bytes.
    """
    bytes = []
    for i,c in enumerate(d):
        byte = ord(c)
        hex_byte = hex(byte)[2:]
        if byte <= 0xf:
            hex_byte = '0' + hex_byte
        if i % 4 == 3:
            hex_byte += ' '
        bytes.append(hex_byte)
    return ''.join(bytes).strip()

def dataToHexUnified(d):
    """ Convert the raw data in 'd' to an hex string with a space every 4 bytes.
    Each 4byte number is prefixed with 0x for easy sed/rx
    Fixme: convert all MC tests to use this routine instead of the above
    """
    bytes = []
    for i,c in enumerate(d):
        byte = ord(c)
        hex_byte = hex(byte)[2:]
        if byte <= 0xf:
            hex_byte = '0' + hex_byte
        if i % 4 == 0:
            hex_byte = '0x' + hex_byte
        if i % 4 == 3:
            hex_byte += ' '
        bytes.append(hex_byte)
    return ''.join(bytes).strip()


def HexDump(valPair):
    """
    1. do not print 'L'
    2. Handle negatives and large numbers by mod (2^numBits)
    3. print fixed length, prepend with zeros.
       Length is exactly 2+(numBits/4)
    4. Do print 0x Why?
       so that they can be easily distinguished using sed/rx
    """
    val, numBits = valPair
    assert 0 <= val < (1 << numBits)

    val = val & (( 1 << numBits) - 1)
    newFmt = "0x%0" + "%d" % (numBits / 4) + "x"
    return newFmt % val
