import lldb


class FacadeExample:
    def __init__(self, bkpt, extra_args, dict):
        self.bkpt = bkpt
        self.extra_args = extra_args
        self.base_sym = None
        self.facade_locs = []
        self.facade_locs_desc = []
        self.cur_facade_loc = 1

        self.sym_name = extra_args.GetValueForKey("symbol").GetStringValue(100)
        self.num_locs = extra_args.GetValueForKey("num_locs").GetIntegerValue(5)
        self.loc_to_miss = extra_args.GetValueForKey("loc_to_miss").GetIntegerValue(
            10000
        )

    def __callback__(self, sym_ctx):
        self.base_sym = sym_ctx.module.FindSymbol(self.sym_name, lldb.eSymbolTypeCode)
        if self.base_sym.IsValid():
            self.bkpt.AddLocation(self.base_sym.GetStartAddress())
            # Locations are 1 based, so to keep things simple, I'm making
            # the array holding locations 1 based as well:
            self.facade_locs_desc.append(
                "This is the zero index, you shouldn't see this"
            )
            self.facade_locs.append(None)
            for i in range(1, self.num_locs + 1):
                self.facade_locs_desc.append(f"Location index: {i}")
                self.facade_locs.append(self.bkpt.AddFacadeLocation())

    def get_short_help(self):
        return f"I am a facade resolver - sym: {self.sym_name} - num_locs: {self.num_locs} - locs_to_miss: {self.loc_to_miss}"

    def was_hit(self, frame, bp_loc):
        tmp_loc = self.cur_facade_loc

        self.cur_facade_loc = self.cur_facade_loc + 1
        if self.cur_facade_loc == self.num_locs + 1:
            self.cur_facade_loc = 1

        if tmp_loc == self.loc_to_miss:
            return None

        return self.facade_locs[tmp_loc]

    def get_location_description(self, bp_loc, desc_level):
        return self.facade_locs_desc[bp_loc.id]
