// RUN: tblgen -I %p %s
def BeforeInclude;

include "Include.inc"

def AfterInclude;

