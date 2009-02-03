-- RUN: %llvmgcc -c %s -I%p/Support
-- RUN: %llvmgcc -c %s -I%p/Support -O2
package body Fat_Fields is
   procedure Proc is
   begin
      if P = null then
         null;
      end if;
   end;
end;
