-- RUN: %llvmgcc -c %s -I%p/Support
package body Global_Constant is
begin
   raise An_Error;
end;
